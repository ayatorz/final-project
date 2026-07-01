from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import cloudscraper
from bs4 import BeautifulSoup
from janome.tokenizer import Tokenizer


DEFAULT_DATASET_PATH = Path("music_artist_dataset_template.csv")
DEFAULT_SOURCES_PATH = Path("lyrics_sources_template.csv")
UTA_NET_BASE_URL = "https://www.uta-net.com"


def normalize_cell(text: str) -> str:
    return " ".join(text.strip().split())


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(handle)]


def save_dataset(path: Path, rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "title", "artist"])
        writer.writeheader()
        writer.writerows(rows)


def normalize_match_text(text: str) -> str:
    return "".join(text.strip().split()).casefold()


@dataclass
class SongSource:
    artist: str
    title: str
    song_url: str = ""


class UtaNetScraper:
    def __init__(self) -> None:
        self.scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "darwin", "desktop": True}
        )

    def search_song_url(self, artist: str, title: str) -> str:
        search_url = f"{UTA_NET_BASE_URL}/search/?Aselect=2&Bselect=3&Keyword={quote(title)}"
        response = self.scraper.get(search_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        expected_title = normalize_match_text(title)
        expected_artist = normalize_match_text(artist)

        for row in soup.select("table.songlist-table tbody.songlist-table-body tr"):
            song_link = row.select_one("td a[href^='/song/'] .songlist-title")
            href_tag = row.select_one("td a[href^='/song/']")
            artist_tag = row.select_one("td:nth-of-type(2) a[href^='/artist/']")
            if not song_link or not href_tag or not artist_tag:
                continue

            row_title = normalize_match_text(song_link.get_text())
            row_artist = normalize_match_text(artist_tag.get_text())
            if row_title == expected_title and row_artist == expected_artist:
                return f"{UTA_NET_BASE_URL}{href_tag['href']}"

        raise LookupError(f"歌詞ネットで '{artist} - {title}' の一致結果が見つかりませんでした。")

    def fetch_lyrics_lines(self, song_url: str) -> list[str]:
        response = self.scraper.get(song_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        lyrics_block = soup.select_one("#kashi_area")
        if lyrics_block is None:
            raise LookupError(f"歌詞本文が見つかりませんでした: {song_url}")

        return [
            normalize_cell(line)
            for line in lyrics_block.get_text("\n").splitlines()
            if normalize_cell(line)
        ]


class LyricsFormatter:
    def __init__(self) -> None:
        self.tokenizer = Tokenizer()

    def tokenize_line(self, line: str) -> str:
        return " ".join(token.surface for token in self.tokenizer.tokenize(line) if token.surface.strip())


def load_song_sources(path: Path) -> list[SongSource]:
    rows = load_csv_rows(path)
    required = {"artist", "title", "song_url"}
    if not rows:
        raise ValueError(f"{path} にデータ行がありません。")
    if set(rows[0]) != required:
        raise ValueError(f"{path} は artist,title,song_url の列が必要です。")

    return [
        SongSource(
            artist=normalize_cell(row["artist"]),
            title=normalize_cell(row["title"]),
            song_url=normalize_cell(row["song_url"]),
        )
        for row in rows
        if normalize_cell(row["artist"]) and normalize_cell(row["title"])
    ]


def append_lyrics(
    dataset_path: Path,
    sources_path: Path,
    *,
    skip_existing_songs: bool,
) -> tuple[int, int]:
    dataset_rows = load_csv_rows(dataset_path)
    sources = load_song_sources(sources_path)
    scraper = UtaNetScraper()
    formatter = LyricsFormatter()

    existing_song_keys = {
        (normalize_match_text(row["artist"]), normalize_match_text(row["title"]))
        for row in dataset_rows
    }
    existing_row_keys = {
        (
            normalize_match_text(row["artist"]),
            normalize_match_text(row["title"]),
            normalize_cell(row["text"]),
        )
        for row in dataset_rows
    }

    appended_rows = 0
    appended_songs = 0

    for source in sources:
        song_key = (normalize_match_text(source.artist), normalize_match_text(source.title))
        if skip_existing_songs and song_key in existing_song_keys:
            print(f"skip existing song: {source.artist} - {source.title}")
            continue

        song_url = source.song_url or scraper.search_song_url(source.artist, source.title)
        lyrics_lines = scraper.fetch_lyrics_lines(song_url)

        new_song_rows = []
        for line in lyrics_lines:
            tokenized_line = formatter.tokenize_line(line)
            row_key = (song_key[0], song_key[1], tokenized_line)
            if not tokenized_line or row_key in existing_row_keys:
                continue
            existing_row_keys.add(row_key)
            new_song_rows.append(
                {
                    "text": tokenized_line,
                    "title": source.title,
                    "artist": source.artist,
                }
            )

        if not new_song_rows:
            print(f"no new rows: {source.artist} - {source.title}")
            continue

        dataset_rows.extend(new_song_rows)
        existing_song_keys.add(song_key)
        appended_rows += len(new_song_rows)
        appended_songs += 1
        print(f"added {len(new_song_rows)} rows: {source.artist} - {source.title}")

    if appended_rows:
        save_dataset(dataset_path, dataset_rows)

    return appended_songs, appended_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape lyrics from Uta-Net and append tokenized rows to the dataset CSV.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES_PATH)
    parser.add_argument(
        "--allow-existing-song-update",
        action="store_true",
        help="Append only missing lines even if the song already exists in the dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    appended_songs, appended_rows = append_lyrics(
        args.dataset,
        args.sources,
        skip_existing_songs=not args.allow_existing_song_update,
    )
    print(f"completed: {appended_songs} songs, {appended_rows} rows added")


if __name__ == "__main__":
    main()
