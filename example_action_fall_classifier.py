"""Simple NumPy baseline for action classification and fall detection.

The example uses the processed TOP_SPRAY release only.  It reads one feature
vector per annotated interval from RD.mat and RA.mat, splits by session to
avoid temporal leakage, trains small one-hidden-layer MLPs, and saves metrics,
predictions, and confusion matrices.

Example:
    py -3.13 examples/example_action_fall_classifier.py \
        --data-root F:\\FALL_RD_RA_RELEASE\\data_v2 \
        --out-dir examples/action_fall_results

The default positive class for binary fall detection is exactly ``Fall down``.
``Fall down and get up`` remains a separate action in the multi-class task and
is treated as a non-fall interval in the binary task.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FPS = 20.0
LABEL_FIELDS = [
    "metadata_id",
    "file_list",
    "temporal_segment_start",
    "temporal_segment_end",
    "metadata",
]
LABEL_ALIASES = {
    "No perosn": "No person",
    "bend posture": "Bend posture",
}
FALL_LABEL = "Fall down"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"F:\FALL_RD_RA_RELEASE\data_v2"),
        help="Release root containing TOP_SPRAY and SIDE_SPRAY.",
    )
    parser.add_argument("--group", default="TOP_SPRAY", choices=["TOP_SPRAY", "SIDE_SPRAY"])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("examples/action_fall_results"),
        help="Directory for matrices, metrics, predictions and model parameters.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--window-frames", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument(
        "--max-intervals-per-session",
        type=int,
        default=0,
        help="Optional cap for a quick smoke test; 0 keeps all intervals.",
    )
    return parser.parse_args()


def normalise_label(value: str) -> str:
    value = " ".join(str(value).strip().split())
    return LABEL_ALIASES.get(value, value)


def read_labels(path: Path) -> List[Tuple[float, float, str]]:
    """Read the two VIA CSV metadata schemas used in the release."""
    rows: List[Tuple[float, float, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        data_lines = [line for line in handle if line.strip() and not line.startswith("#")]
    reader = csv.DictReader(data_lines, fieldnames=LABEL_FIELDS)
    for row in reader:
        try:
            metadata = json.loads(row["metadata"])
            if not isinstance(metadata, dict):
                continue
            raw_label = metadata.get("Activity") or metadata.get("TEMPORAL-SEGMENTS")
            if not raw_label:
                continue
            start = float(row["temporal_segment_start"])
            end = float(row["temporal_segment_end"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        rows.append((start, end, normalise_label(raw_label)))
    return rows


def read_release_arrays(session_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return RD magnitude and RA power as [range, Doppler/angle, frame]."""
    with h5py.File(session_dir / "RD.mat", "r") as rd_file:
        stored = rd_file["RD"][()]
        if stored.dtype.names != ("real", "imag"):
            raise TypeError(f"Expected compound complex RD in {session_dir}")
        rd = (stored["real"] + 1j * stored["imag"]).astype(np.complex64)
    with h5py.File(session_dir / "RA.mat", "r") as ra_file:
        ra = np.asarray(ra_file["RA"][()], dtype=np.float32)
    # MATLAB storage is [frame, Doppler/angle, range].
    return np.abs(rd).transpose(2, 1, 0), ra.transpose(2, 1, 0)


def block_pool(matrix: np.ndarray, out_rows: int, out_cols: int) -> np.ndarray:
    """Average-pool a 2-D range x Doppler/angle map to a small fixed grid."""
    row_blocks = np.array_split(matrix, out_rows, axis=0)
    pooled_rows = []
    for row_block in row_blocks:
        col_blocks = np.array_split(row_block, out_cols, axis=1)
        pooled_rows.append(np.stack([block.mean() for block in col_blocks]))
    return np.stack(pooled_rows)


def interval_feature(
    rd: np.ndarray,
    ra: np.ndarray,
    start_s: float,
    end_s: float,
    window_frames: int,
) -> np.ndarray:
    """Create a compact temporal feature from an annotated interval center."""
    n_frames = rd.shape[2]
    center = int(round(((start_s + end_s) * 0.5) * FPS))
    center = max(0, min(n_frames - 1, center))
    half = max(1, window_frames // 2)
    left = max(0, center - half)
    right = min(n_frames, left + max(2, window_frames))
    left = max(0, right - max(2, window_frames))

    return window_feature(rd[:, :, left:right], ra[:, :, left:right])


def window_feature(rd_window: np.ndarray, ra_window: np.ndarray) -> np.ndarray:
    """Pool one already-loaded temporal window into a fixed-size feature."""
    rd_window = np.log1p(rd_window).astype(np.float32)
    ra_window = np.log1p(np.maximum(ra_window, 0.0)).astype(np.float32)
    # Pool the spatial maps before temporal aggregation.  This keeps the demo
    # small enough to run on a laptop while retaining range and motion/angle
    # structure from both released representations.
    # The release dimensions permit a fast fixed-grid pooling path: 39 of the
    # 51 range bins, all 128 Doppler bins, and 27 of the 33 angle bins form
    # integer blocks. The remaining bins are outside this compact baseline.
    if rd_window.shape[0] >= 39 and rd_window.shape[1] >= 128 and ra_window.shape[0] >= 39 and ra_window.shape[1] >= 27:
        rd_stack = rd_window[:39, :128, :].transpose(2, 0, 1).reshape(rd_window.shape[2], 13, 3, 16, 8).mean(axis=(2, 4))
        ra_stack = ra_window[:39, :27, :].transpose(2, 0, 1).reshape(ra_window.shape[2], 13, 3, 9, 3).mean(axis=(2, 4))
    else:
        rd_stack = np.stack([block_pool(rd_window[:, :, i], 13, 16) for i in range(rd_window.shape[2])])
        ra_stack = np.stack([block_pool(ra_window[:, :, i], 13, 9) for i in range(ra_window.shape[2])])
    feature = np.concatenate(
        [rd_stack.mean(axis=0).ravel(), rd_stack.std(axis=0).ravel(),
         ra_stack.mean(axis=0).ravel(), ra_stack.std(axis=0).ravel()]
    ).astype(np.float32)
    return feature


def collect_sessions(group_dir: Path, window_frames: int, cap: int) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[dict]]:
    features: List[np.ndarray] = []
    labels: List[str] = []
    session_names: List[str] = []
    interval_ids: List[str] = []
    session_records: List[dict] = []
    for session_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        label_path = session_dir / "label.csv"
        intervals = read_labels(label_path)
        if not intervals or not (session_dir / "RD.mat").exists() or not (session_dir / "RA.mat").exists():
            continue
        if cap > 0:
            intervals = intervals[:cap]
        try:
            rd_file = h5py.File(session_dir / "RD.mat", "r")
            ra_file = h5py.File(session_dir / "RA.mat", "r")
            rd_dataset = rd_file["RD"]
            ra_dataset = ra_file["RA"]
            if rd_dataset.dtype.names != ("real", "imag"):
                raise TypeError(f"Expected compound complex RD in {session_dir}")
            n_frames = int(rd_dataset.shape[0])
        except (OSError, ValueError, TypeError) as exc:
            print(f"[skip] {session_dir.name}: {exc}")
            continue
        try:
            n_before = len(features)
            for interval_index, (start, end, label) in enumerate(intervals):
                center = int(round(((start + end) * 0.5) * FPS))
                center = max(0, min(n_frames - 1, center))
                half = max(1, window_frames // 2)
                left = max(0, center - half)
                right = min(n_frames, left + max(2, window_frames))
                left = max(0, right - max(2, window_frames))
                stored = rd_dataset[left:right]
                rd_window = (stored["real"] + 1j * stored["imag"]).astype(np.complex64).transpose(2, 1, 0)
                ra_window = np.asarray(ra_dataset[left:right], dtype=np.float32).transpose(2, 1, 0)
                features.append(window_feature(np.abs(rd_window), ra_window))
                labels.append(label)
                session_names.append(session_dir.name)
                interval_ids.append(f"{session_dir.name}:{interval_index:04d}")
            session_records.append({
                "session_id": session_dir.name,
                "frame_count": n_frames,
                "interval_count": len(features) - n_before,
            })
            print(f"[read] {session_dir.name}: {len(features) - n_before} intervals, {n_frames} frames")
        finally:
            rd_file.close()
            ra_file.close()
    if not features:
        raise RuntimeError(f"No labelled sessions found under {group_dir}")
    return np.stack(features), np.asarray(labels), session_names, interval_ids, session_records


def split_by_session(session_names: Sequence[str], labels: Sequence[str], test_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    unique_sessions = np.asarray(sorted(set(session_names)))
    if len(unique_sessions) < 2:
        raise ValueError("At least two labelled sessions are required for a session split")
    rng = np.random.default_rng(seed)
    shuffled = unique_sessions.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    test_sessions = list(shuffled[-n_test:])
    train_sessions = list(shuffled[:-n_test])
    all_labels = set(labels)
    # Try to keep every class in both partitions when the session distribution allows it.
    for cls in sorted(all_labels):
        if not any(cls == y for s, y in zip(session_names, labels) if s in test_sessions):
            candidates = [s for s in train_sessions if any(y == cls and ss == s for ss, y in zip(session_names, labels))]
            if candidates:
                moved = candidates[0]
                train_sessions.remove(moved)
                test_sessions.append(moved)
    train_mask = np.asarray([s in train_sessions for s in session_names])
    test_mask = np.asarray([s in test_sessions for s in session_names])
    return train_mask, test_mask, sorted(train_sessions), sorted(test_sessions)


class NumpyMLP:
    def __init__(self, input_size: int, hidden_size: int, class_count: int, seed: int):
        rng = np.random.default_rng(seed)
        self.w1 = (rng.standard_normal((input_size, hidden_size)).astype(np.float32) * np.sqrt(2.0 / input_size))
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.w2 = (rng.standard_normal((hidden_size, class_count)).astype(np.float32) * np.sqrt(2.0 / hidden_size))
        self.b2 = np.zeros(class_count, dtype=np.float32)

    def logits(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hidden = np.maximum(0.0, x @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2, hidden

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(x)[0], axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, learning_rate: float, seed: int, class_weights: np.ndarray) -> List[float]:
        rng = np.random.default_rng(seed)
        history: List[float] = []
        n = len(x)
        for epoch in range(epochs):
            order = rng.permutation(n)
            total_loss = 0.0
            for begin in range(0, n, batch_size):
                idx = order[begin:begin + batch_size]
                xb, yb = x[idx], y[idx]
                scores, hidden = self.logits(xb)
                scores = scores - scores.max(axis=1, keepdims=True)
                exp_scores = np.exp(scores)
                probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                weights = class_weights[yb]
                loss = -np.sum(weights * np.log(probs[np.arange(len(yb)), yb] + 1e-8)) / max(1, len(yb))
                total_loss += float(loss) * len(yb)
                ds = probs
                ds[np.arange(len(yb)), yb] -= 1.0
                ds *= (weights / max(1, len(yb)))[:, None]
                dw2 = hidden.T @ ds
                db2 = ds.sum(axis=0)
                dh = (ds @ self.w2.T) * (hidden > 0)
                dw1 = xb.T @ dh
                db1 = dh.sum(axis=0)
                self.w2 -= learning_rate * dw2
                self.b2 -= learning_rate * db2
                self.w1 -= learning_rate * dw1
                self.b1 -= learning_rate * db1
            history.append(total_loss / max(1, n))
            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
                print(f"[train] epoch {epoch + 1:03d}/{epochs}: loss={history[-1]:.4f}")
        return history


def metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]) -> dict:
    result = {"accuracy": float(np.mean(y_true == y_pred)), "classes": {}}
    f1_values = []
    for i, name in enumerate(class_names):
        tp = int(np.sum((y_true == i) & (y_pred == i)))
        fp = int(np.sum((y_true != i) & (y_pred == i)))
        fn = int(np.sum((y_true == i) & (y_pred != i)))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        f1_values.append(f1)
        result["classes"][name] = {"support": int(np.sum(y_true == i)), "precision": precision, "recall": recall, "f1": f1}
    result["macro_f1"] = float(np.mean(f1_values)) if f1_values else 0.0
    return result


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str], path: Path, title: str) -> None:
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    np.savetxt(path.with_suffix(".csv"), matrix, fmt="%d", delimiter=",")
    fig_size = max(7.0, 0.55 * len(class_names) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > threshold else "black", fontsize=8)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_task(
    name: str,
    x_train: np.ndarray,
    y_train_text: np.ndarray,
    x_test: np.ndarray,
    y_test_text: np.ndarray,
    out_dir: Path,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict:
    class_names = sorted(set(y_train_text) | set(y_test_text))
    class_to_id = {name: i for i, name in enumerate(class_names)}
    y_train = np.asarray([class_to_id[y] for y in y_train_text], dtype=np.int64)
    y_test = np.asarray([class_to_id[y] for y in y_test_text], dtype=np.int64)
    counts = np.bincount(y_train, minlength=len(class_names)).astype(np.float32)
    class_weights = counts.sum() / np.maximum(1.0, counts * len(class_names))
    model = NumpyMLP(x_train.shape[1], hidden_size, len(class_names), seed)
    history = model.fit(x_train, y_train, epochs, batch_size, learning_rate, seed, class_weights)
    prediction = model.predict(x_test)
    result = metrics(y_test, prediction, class_names)
    result.update({"task": name, "class_to_id": class_to_id, "train_samples": len(y_train), "test_samples": len(y_test), "loss_history": history})
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / f"{name}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", "predicted_label"])
        writer.writerows((y_test_text[i], class_names[prediction[i]]) for i in range(len(prediction)))
    save_confusion_matrix(y_test, prediction, class_names, out_dir / f"{name}_confusion_matrix.png", name.replace("_", " ").title())
    np.savez_compressed(out_dir / f"{name}_model.npz", w1=model.w1, b1=model.b1, w2=model.w2, b2=model.b2, mean=x_train.mean(axis=0), std=x_train.std(axis=0) + 1e-6)
    print(f"[{name}] accuracy={result['accuracy']:.3f}, macro-F1={result['macro_f1']:.3f}")
    return result


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.test_fraction < 0.5:
        raise ValueError("--test-fraction should be between 0.05 and 0.49")
    group_dir = args.data_root / args.group
    args.out_dir.mkdir(parents=True, exist_ok=True)
    x, labels, session_names, interval_ids, session_records = collect_sessions(group_dir, args.window_frames, args.max_intervals_per_session)
    train_mask, test_mask, train_sessions, test_sessions = split_by_session(session_names, labels, args.test_fraction, args.seed)
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError("Session split produced an empty train or test partition")
    mean = x[train_mask].mean(axis=0)
    std = x[train_mask].std(axis=0) + 1e-6
    x_scaled = ((x - mean) / std).astype(np.float32)
    np.savez_compressed(args.out_dir / "feature_standardization.npz", mean=mean, std=std)
    action_result = run_task(
        "action_classification",
        x_scaled[train_mask], labels[train_mask], x_scaled[test_mask], labels[test_mask],
        args.out_dir, args.hidden_size, args.epochs, args.batch_size, args.learning_rate, args.seed,
    )
    binary_labels = np.where(labels == FALL_LABEL, "Fall down", "Non-fall")
    fall_result = run_task(
        "fall_detection",
        x_scaled[train_mask], binary_labels[train_mask], x_scaled[test_mask], binary_labels[test_mask],
        args.out_dir, args.hidden_size, args.epochs, args.batch_size, args.learning_rate, args.seed + 1,
    )
    split_info = {
        "data_root": str(args.data_root),
        "group": args.group,
        "fps": FPS,
        "feature": "temporal mean/std of pooled log1p(RD magnitude) and log1p(RA power)",
        "window_frames": args.window_frames,
        "train_sessions": train_sessions,
        "test_sessions": test_sessions,
        "session_count": len(session_records),
        "interval_count": int(len(labels)),
        "action_labels": sorted(set(labels)),
        "fall_positive_label": FALL_LABEL,
        "note": "Fall down and get up is retained as a separate action and is non-fall for binary detection.",
        "action_metrics": action_result,
        "fall_metrics": fall_result,
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    print(f"[done] outputs written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
