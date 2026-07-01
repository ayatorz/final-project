from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


DATASET_PATH = Path("music_artist_dataset_template.csv")
OUTPUT_DIR = Path("outputs")
RANDOM_SEED = 42

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_dataset(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {"text", "title", "artist"}
    if not rows:
        raise ValueError(f"{path} has no data rows.")
    if set(rows[0]) != required:
        raise ValueError(f"{path} must have columns: text,title,artist")

    clean_rows = []
    for row in rows:
        text = normalize_text(row["text"])
        title = normalize_text(row["title"])
        artist = normalize_text(row["artist"])
        if text and title and artist:
            clean_rows.append({"text": text, "title": title, "artist": artist})

    if not clean_rows:
        raise ValueError(f"{path} has no usable rows.")
    return clean_rows


def build_inputs(rows: list[dict[str, str]], use_title: bool) -> tuple[list[str], list[str]]:
    if use_title:
        x = [f"{row['title']} [TITLE] {row['text']}" for row in rows]
    else:
        x = [row["text"] for row in rows]
    y = [row["artist"] for row in rows]
    return x, y


def split_dataset_by_song(
    rows: list[dict[str, str]],
    *,
    train_songs: int = 3,
    valid_songs: int = 1,
    test_songs: int = 1,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, object]]:
    rows_by_artist_title: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        rows_by_artist_title[row["artist"]][row["title"]].append(row)

    splits = {"train": [], "valid": [], "test": []}
    song_split_summary = {}
    rng = random.Random(RANDOM_SEED)

    required_songs = train_songs + valid_songs + test_songs
    for artist, rows_by_title in sorted(rows_by_artist_title.items()):
        titles = sorted(rows_by_title)
        if len(titles) < required_songs:
            raise ValueError(
                f"{artist} needs at least {required_songs} songs, but has {len(titles)}."
            )

        rng.shuffle(titles)
        train_titles = titles[:train_songs]
        valid_titles = titles[train_songs : train_songs + valid_songs]
        test_titles = titles[train_songs + valid_songs : train_songs + valid_songs + test_songs]

        song_split_summary[artist] = {
            "train": train_titles,
            "valid": valid_titles,
            "test": test_titles,
        }

        for split_name, split_titles in song_split_summary[artist].items():
            for title in split_titles:
                splits[split_name].extend(rows_by_title[title])

    summary = {
        "strategy": "stratified by artist, grouped by song title",
        "seed": RANDOM_SEED,
        "song_counts_per_artist": {
            "train": train_songs,
            "valid": valid_songs,
            "test": test_songs,
        },
        "songs": song_split_summary,
        "row_counts": {name: len(split_rows) for name, split_rows in splits.items()},
        "artist_row_counts": {
            name: dict(Counter(row["artist"] for row in split_rows))
            for name, split_rows in splits.items()
        },
    }
    return splits, summary


def save_split_csvs(splits: dict[str, list[dict[str, str]]], output_dir: Path) -> None:
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["text", "title", "artist"]
    for split_name, split_rows in splits.items():
        with (split_dir / f"{split_name}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)


def collect_artist_keywords(
    rows: list[dict[str, str]],
    *,
    limit: int,
    max_features: int,
) -> dict[str, list[dict[str, object]]]:
    texts_by_artist: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        texts_by_artist[row["artist"]].append(row["text"])

    artists = sorted(texts_by_artist)
    artist_documents = [" ".join(texts_by_artist[artist]) for artist in artists]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\S+",
        ngram_range=(1, 1),
        max_features=max_features,
        lowercase=False,
        sublinear_tf=True,
    )
    scores = vectorizer.fit_transform(artist_documents).toarray()
    words = np.array(vectorizer.get_feature_names_out())

    keywords: dict[str, list[dict[str, object]]] = {}
    for artist, artist_scores in zip(artists, scores):
        ranked_indices = np.argsort(artist_scores)[::-1]
        top_keywords = []
        for index in ranked_indices:
            score = float(artist_scores[index])
            if score <= 0:
                break
            top_keywords.append({"word": str(words[index]), "score": score})
            if len(top_keywords) >= limit:
                break
        keywords[artist] = top_keywords

    return keywords


def collect_artist_accuracy(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, dict[str, object]]:
    results = {}
    labels = sorted(set(y_true))
    for label in labels:
        total = sum(true_label == label for true_label in y_true)
        correct = sum(
            true_label == label and predicted_label == label
            for true_label, predicted_label in zip(y_true, y_pred)
        )
        results[label] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        }
    return results


def collect_confusion_summary(
    labels: list[str],
    matrix: list[list[int]],
) -> dict[str, dict[str, int]]:
    summary = {}
    for true_label, row in zip(labels, matrix):
        predicted_counts = {}
        for predicted_label, count in zip(labels, row):
            if count:
                predicted_counts[predicted_label] = int(count)
        summary[true_label] = predicted_counts
    return summary


def evaluate_song_majority_vote(
    classifier: Pipeline,
    rows: list[dict[str, str]],
    *,
    use_title: bool,
) -> dict[str, object]:
    x, _ = build_inputs(rows, use_title)
    predictions = classifier.predict(x)

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row, predicted_label in zip(rows, predictions):
        grouped[(row["artist"], row["title"])].append(str(predicted_label))

    song_results = []
    correct = 0
    for (true_artist, title), predicted_labels in sorted(grouped.items()):
        vote_counts = Counter(predicted_labels)
        predicted_artist = sorted(
            vote_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        is_correct = predicted_artist == true_artist
        if is_correct:
            correct += 1
        song_results.append(
            {
                "title": title,
                "true_artist": true_artist,
                "predicted_artist": predicted_artist,
                "correct": is_correct,
                "line_count": len(predicted_labels),
                "votes": dict(sorted(vote_counts.items())),
            }
        )

    total = len(song_results)
    return {
        "song_count": total,
        "correct_songs": correct,
        "song_accuracy": correct / total if total else 0.0,
        "songs": song_results,
    }


def train_classifier(
    splits: dict[str, list[dict[str, str]]],
    *,
    use_title: bool,
    max_iter: int,
    max_features: int,
) -> tuple[Pipeline, dict[str, object]]:
    x_train, y_train = build_inputs(splits["train"], use_title)
    x_valid, y_valid = build_inputs(splits["valid"], use_title)
    x_test, y_test = build_inputs(splits["test"], use_title)
    all_labels = y_train + y_valid + y_test

    model = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer="word",
                    token_pattern=r"(?u)\S+",
                    ngram_range=(1, 2),
                    max_features=max_features,
                    lowercase=False,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-2,
                    batch_size=32,
                    learning_rate_init=1e-4,
                    max_iter=max_iter,
                    random_state=RANDOM_SEED,
                    early_stopping=False,
                    n_iter_no_change=15,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    labels = sorted(set(all_labels))
    valid_pred = model.predict(x_valid)
    test_pred = model.predict(x_test)
    test_pred_labels = [str(label) for label in test_pred]
    test_confusion_matrix = confusion_matrix(y_test, test_pred, labels=labels).tolist()

    metrics: dict[str, object] = {
        "dataset_rows": sum(len(split_rows) for split_rows in splits.values()),
        "train_rows": len(x_train),
        "valid_rows": len(x_valid),
        "test_rows": len(x_test),
        "use_title": use_title,
        "vectorizer": "word 1-2gram TF-IDF",
        "classifier": "MLPClassifier hidden_layer_sizes=(64, 32)",
        "artist_counts": dict(Counter(all_labels)),
        "valid_accuracy": accuracy_score(y_valid, valid_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "labels": labels,
        "artist_accuracy": collect_artist_accuracy(y_test, test_pred_labels),
        "confusion_matrix": test_confusion_matrix,
        "confusion_summary": collect_confusion_summary(labels, test_confusion_matrix),
        "classification_report": classification_report(
            y_test,
            test_pred,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
        "misclassified_examples": collect_misclassified(x_test, y_test, test_pred),
    }
    return model, metrics


def collect_misclassified(
    x_test: list[str], y_test: list[str], y_pred: np.ndarray, limit: int = 12
) -> list[dict[str, str]]:
    examples = []
    for text, true_label, predicted_label in zip(x_test, y_test, y_pred):
        if true_label != predicted_label:
            examples.append(
                {
                    "true": true_label,
                    "predicted": str(predicted_label),
                    "input": text[:120],
                }
            )
        if len(examples) >= limit:
            break
    return examples


@dataclass
class WordNgramGenerator:
    order: int = 2

    def __post_init__(self) -> None:
        self.table: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.starts: list[tuple[str, ...]] = []

    def fit(self, texts: list[str]) -> None:
        pad = ["<BOS>"] * self.order
        for text in texts:
            words = normalize_text(text).split()
            if not words:
                continue
            sequence = pad + words + ["<EOS>"]
            self.starts.append(tuple(sequence[: self.order]))
            for i in range(len(sequence) - self.order):
                key = tuple(sequence[i : i + self.order])
                nxt = sequence[i + self.order]
                self.table[key][nxt] += 1

    def generate(self, seed: str, *, length: int, temperature: float) -> str:
        if not self.table:
            raise ValueError("generator has not been fitted")

        seed_words = normalize_text(seed).split()
        state_words = (["<BOS>"] * self.order + seed_words)[-self.order :]
        state = tuple(state_words)
        output = seed_words[:]

        for _ in range(length):
            counter = self.table.get(state)
            if not counter:
                state = random.choice(self.starts)
                counter = self.table[state]

            words = list(counter)
            weights = np.array([counter[word] for word in words], dtype=np.float64)
            weights = np.power(weights, 1.0 / max(temperature, 1e-6))
            weights = weights / weights.sum()
            nxt = random.choices(words, weights=weights, k=1)[0]

            if nxt == "<EOS>":
                break

            output.append(nxt)
            state = tuple((list(state) + [nxt])[-self.order :])

        return " ".join(output).strip()


def train_generators(rows: list[dict[str, str]], order: int) -> dict[str, WordNgramGenerator]:
    texts_by_artist: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        texts_by_artist[row["artist"]].append(row["text"])

    generators = {}
    for artist, texts in texts_by_artist.items():
        generator = WordNgramGenerator(order=order)
        generator.fit(texts)
        generators[artist] = generator
    return generators


def predict_generated_texts(
    classifier: Pipeline,
    generated: dict[str, list[str]],
) -> list[dict[str, object]]:
    results = []
    classes = list(classifier.classes_)
    for target_artist, texts in generated.items():
        for text in texts:
            proba = classifier.predict_proba([text])[0]
            ranked = sorted(zip(classes, proba), key=lambda item: item[1], reverse=True)
            results.append(
                {
                    "target_artist": target_artist,
                    "text": text,
                    "predicted_artist": ranked[0][0],
                    "probabilities": {artist: float(score) for artist, score in ranked},
                }
            )
    return results


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    rows = load_dataset(args.dataset)
    OUTPUT_DIR.mkdir(exist_ok=True)
    splits, split_summary = split_dataset_by_song(
        rows,
        train_songs=args.train_songs,
        valid_songs=args.valid_songs,
        test_songs=args.test_songs,
    )
    save_split_csvs(splits, OUTPUT_DIR)
    save_json(OUTPUT_DIR / "split_summary.json", split_summary)

    artist_keywords = collect_artist_keywords(
        splits["train"],
        limit=args.keyword_count,
        max_features=args.max_features,
    )
    save_json(OUTPUT_DIR / "artist_keywords.json", artist_keywords)

    classifier, metrics = train_classifier(
        splits,
        use_title=args.use_title,
        max_iter=args.max_iter,
        max_features=args.max_features,
    )
    with (OUTPUT_DIR / "artist_classifier.pkl").open("wb") as f:
        pickle.dump(classifier, f)
    save_json(OUTPUT_DIR / "classification_metrics.json", metrics)

    song_vote_metrics = None
    if args.song_vote:
        song_vote_metrics = evaluate_song_majority_vote(
            classifier,
            splits["test"],
            use_title=args.use_title,
        )
        save_json(OUTPUT_DIR / "song_vote_metrics.json", song_vote_metrics)

    # Text generation is intentionally disabled for this classification-focused version.
    # generators = train_generators(splits["train"], order=args.order)
    # generated: dict[str, list[str]] = {}
    # for artist, generator in generators.items():
    #     generated[artist] = [
    #         generator.generate(args.seed_text, length=args.generate_length, temperature=args.temperature)
    #         for _ in range(args.samples)
    #     ]
    # save_json(OUTPUT_DIR / "generated_texts.json", generated)
    #
    # generated_predictions = predict_generated_texts(classifier, generated)
    # save_json(OUTPUT_DIR / "generated_predictions.json", generated_predictions)

    print("=== Artist classifier ===")
    print(f"rows: {metrics['dataset_rows']}")
    print(
        "split rows:",
        f"train={metrics['train_rows']}",
        f"valid={metrics['valid_rows']}",
        f"test={metrics['test_rows']}",
    )
    print(f"valid accuracy: {metrics['valid_accuracy']:.3f}")
    print(f"test accuracy: {metrics['test_accuracy']:.3f}")
    print("labels:", ", ".join(metrics["labels"]))
    print()
    print("=== Artist accuracy ===")
    for artist, result in metrics["artist_accuracy"].items():
        print(
            f"{artist}:",
            f"{result['accuracy']:.3f}",
            f"({result['correct']}/{result['total']})",
        )
    print()
    print("=== Confusion summary ===")
    for true_artist, predicted_counts in metrics["confusion_summary"].items():
        counts = ", ".join(
            f"{predicted_artist}={count}"
            for predicted_artist, count in predicted_counts.items()
        )
        print(f"{true_artist} -> {counts}")
    print()
    if song_vote_metrics is not None:
        print("=== Song majority vote ===")
        print(
            "song accuracy:",
            f"{song_vote_metrics['song_accuracy']:.3f}",
            f"({song_vote_metrics['correct_songs']}/{song_vote_metrics['song_count']})",
        )
        for song in song_vote_metrics["songs"]:
            mark = "OK" if song["correct"] else "NG"
            print(
                f"{mark} {song['title']}:",
                f"true={song['true_artist']}",
                f"predicted={song['predicted_artist']}",
            )
        print()
    print("=== Song split ===")
    for artist, artist_splits in split_summary["songs"].items():
        print(
            f"{artist}:",
            f"train={', '.join(artist_splits['train'])}",
            f"valid={', '.join(artist_splits['valid'])}",
            f"test={', '.join(artist_splits['test'])}",
        )
    print()
    print("=== Artist keywords ===")
    for artist, keywords in artist_keywords.items():
        top_words = [str(item["word"]) for item in keywords]
        print(f"{artist}: {', '.join(top_words)}")
    print()
    # print("=== Generated samples ===")
    # for item in generated_predictions[: args.samples * len(generators)]:
    #     print(f"[{item['target_artist']}] -> predicted {item['predicted_artist']}: {item['text']}")
    # print()
    print(f"saved: {OUTPUT_DIR / 'artist_classifier.pkl'}")
    print(f"saved: {OUTPUT_DIR / 'split_summary.json'}")
    print(f"saved: {OUTPUT_DIR / 'splits'}")
    print(f"saved: {OUTPUT_DIR / 'classification_metrics.json'}")
    print(f"saved: {OUTPUT_DIR / 'artist_keywords.json'}")
    if song_vote_metrics is not None:
        print(f"saved: {OUTPUT_DIR / 'song_vote_metrics.json'}")
    # print(f"saved: {OUTPUT_DIR / 'generated_texts.json'}")
    # print(f"saved: {OUTPUT_DIR / 'generated_predictions.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a word-based neural-network artist classifier and analyze artist predictions.",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--use-title", action="store_true", help="Add song title to classifier input.")
    parser.add_argument("--train-songs", type=int, default=3, help="Songs per artist for training.")
    parser.add_argument("--valid-songs", type=int, default=1, help="Songs per artist for validation.")
    parser.add_argument("--test-songs", type=int, default=1, help="Songs per artist for testing.")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--keyword-count", type=int, default=15, help="Important words to show per artist.")
    parser.add_argument("--song-vote", action="store_true", help="Evaluate test songs by majority vote.")
    # Generation options are disabled in this classification-focused version.
    # parser.add_argument("--order", type=int, default=2, help="Word n-gram order for generation.")
    # parser.add_argument("--seed-text", default="夜", help="Beginning text for generation.")
    # parser.add_argument("--generate-length", type=int, default=12, help="Maximum number of generated words.")
    # parser.add_argument("--temperature", type=float, default=0.9)
    # parser.add_argument("--samples", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
