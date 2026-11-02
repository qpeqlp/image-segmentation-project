"""
evaluate.py
-----------
ประเมินผลโมเดล best.pt และสร้างไฟล์ผลลัพธ์ในโฟลเดอร์ results/

    python src/evaluate.py

ไฟล์ที่สร้าง
- results/results.png                 กราฟ Learning Curve จาก results.txt (ผลการ Train 100 epochs)
- results/confusion_matrix.png        Confusion Matrix (Validation + Test) แบบ IoU >= 0.5
- results/detection_result.jpg        ภาพตัวอย่างผลตรวจจับชุด Test (t1-t5)
- results/preprocessing_comparison.png  ภาพเปรียบเทียบเทคนิค Image Processing
- results/preprocessing_eval.csv      ความแม่นยำเมื่อใช้ Preprocessing แต่ละแบบ
- results/metrics.json                สรุปตัวเลขทั้งหมด
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection import DefectDetector  # noqa: E402
from src import preprocessing as pp  # noqa: E402

DATASET = ROOT / "dataset"
RESULTS = ROOT / "results"
CLASSES = ["damaged", "intact"]

SPLITS = {
    "train": [f"p{i}.JPG" for i in range(1, 125)],
    "val": [f"p{i}.JPG" for i in range(200, 239)],
    "test": [f"t{i}.JPG" for i in range(1, 6)],
}


# ---------------------------------------------------------------------------
def load_ground_truth() -> dict[str, list[tuple[int, list[float]]]]:
    """อ่าน jarlids_annots.csv -> {filename: [(class_id, [x1,y1,x2,y2]), ...]}"""
    df = pd.read_csv(DATASET / "jarlids_annots.csv")
    gt: dict[str, list] = {}
    for _, r in df.iterrows():
        attr = json.loads(r["region_attributes"])
        if "type" not in attr:
            continue
        s = json.loads(r["region_shape_attributes"])
        box = [s["x"], s["y"], s["x"] + s["width"], s["y"] + s["height"]]
        gt.setdefault(r["filename"], []).append((CLASSES.index(attr["type"]), box))
    return gt


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(preds, gts, thr=0.5):
    """
    จับคู่ผลทำนายกับ Ground Truth (greedy ตาม confidence)
    คืน confusion matrix ขนาด 3x3: แถว = ทำนาย (damaged, intact, background), คอลัมน์ = จริง
    """
    cm = np.zeros((3, 3), dtype=int)
    used = set()
    for d in sorted(preds, key=lambda d: -d.confidence):
        best, bj = 0.0, -1
        for j, (_, g) in enumerate(gts):
            if j in used:
                continue
            v = iou((d.x1, d.y1, d.x2, d.y2), g)
            if v > best:
                best, bj = v, j
        if best >= thr:
            used.add(bj)
            cm[d.class_id, gts[bj][0]] += 1
        else:
            cm[d.class_id, 2] += 1  # False Positive (ไม่มีวัตถุจริง)
    for j, (c, _) in enumerate(gts):
        if j not in used:
            cm[2, c] += 1  # False Negative (ตรวจไม่พบ)
    return cm


def metrics_from_cm(cm) -> dict:
    out = {}
    for i, name in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[i, :].sum() - tp
        fn = cm[:, i].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out[name] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
                     "tp": int(tp), "fp": int(fp), "fn": int(fn)}
    n_obj = cm[:, :2].sum()
    out["accuracy_all_objects"] = round(float(np.trace(cm[:2, :2]) / n_obj), 3) if n_obj else 0.0
    return out


def run_split(det, files, gt, **opts):
    cm = np.zeros((3, 3), dtype=int)
    for f in files:
        img = cv2.imread(str(DATASET / f))
        if img is None:
            continue
        cm += match(det.detect(img, **opts).detections, gt.get(f, []))
    return cm


# ---------------------------------------------------------------------------
def plot_confusion(cm, path, title):
    labels = CLASSES + ["background"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    norm = cm / np.maximum(cm.sum(axis=0, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            if i == 2 and j == 2:
                continue
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.2f})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "#222", fontsize=11)
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("True (ground truth)")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(results_txt: Path, path: Path):
    """results.txt ของ YOLOv7: epoch, mem, box, obj, cls, total, labels, img_size, P, R, mAP.5, mAP.5:.95, val_box, val_obj, val_cls"""
    rows = [ln.split() for ln in results_txt.read_text().strip().splitlines()]
    cols = ["epoch", "mem", "box", "obj", "cls", "total", "labels", "img", "P", "R",
            "mAP50", "mAP50_95", "val_box", "val_obj", "val_cls"]
    df = pd.DataFrame(rows, columns=cols[: len(rows[0])])
    df["epoch"] = df["epoch"].str.split("/").str[0].astype(int)
    for c in cols[2:]:
        df[c] = df[c].astype(float)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    spec = [("box", "val_box", "Box loss"), ("obj", "val_obj", "Objectness loss"),
            ("cls", "val_cls", "Classification loss"), ("P", None, "Precision"),
            ("R", None, "Recall"), ("mAP50", "mAP50_95", "mAP")]
    for ax, (a, b, t) in zip(axes.flat, spec):
        ax.plot(df["epoch"], df[a], color="#2a6fdb", lw=1.6,
                label="train" if b and "val" in b else a)
        if b:
            ax.plot(df["epoch"], df[b], color="#e8590c", lw=1.6,
                    label="val" if "val" in b else "mAP@0.5:0.95")
            ax.legend(frameon=False)
        ax.set_title(t)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    fig.suptitle(f"YOLOv7 training curves – jar lid defect detection (epoch 0–{int(df['epoch'].max())}, best.pt checkpoint)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

    best = df.loc[df["mAP50"].idxmax()]
    return {"epochs": int(df["epoch"].max()) + 1, "best_epoch": int(best["epoch"]),
            "best_val_mAP50": round(float(best["mAP50"]), 3),
            "best_val_mAP50_95": round(float(best["mAP50_95"]), 3),
            "last_precision": round(float(df["P"].iloc[-1]), 3),
            "last_recall": round(float(df["R"].iloc[-1]), 3)}


def make_test_grid(det, path):
    tiles = []
    for f in SPLITS["test"]:
        img = cv2.imread(str(DATASET / f))
        res = det.detect(img)
        t = det.draw(img, res)
        cv2.putText(t, f, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(t, f, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(t, (640, 480)))
    tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in (0, 3)])
    cv2.imwrite(str(path), grid)


PREPROCESS_VARIANTS = {
    "original": {},
    "CLAHE": {"use_clahe": True},
    "denoise": {"use_denoise": True},
    "sharpen": {"use_sharpen": True},
    "glare_reduction": {"use_glare": True},
    "brightness+20": {"brightness": 20},
    "contrast x1.3": {"contrast": 1.3},
}


def make_preprocessing_figure(path):
    img = cv2.imread(str(DATASET / "p15.JPG"))
    shown = ["original", "CLAHE", "denoise", "sharpen", "glare_reduction", "contrast x1.3"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    for ax, k in zip(axes.flat, shown):
        ax.imshow(pp.bgr_to_rgb(pp.enhance(img, **PREPROCESS_VARIANTS[k])))
        ax.set_title(k)
        ax.axis("off")
    fig.suptitle("OpenCV preprocessing experiments (p15.JPG)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    RESULTS.mkdir(exist_ok=True)
    det = DefectDetector()
    gt = load_ground_truth()
    summary: dict = {"classes": CLASSES, "conf_thres": det.conf_thres, "iou_match": 0.5}

    if (RESULTS / "results.txt").exists():
        summary["training"] = plot_training_curves(RESULTS / "results.txt", RESULTS / "results.png")
        print("training:", summary["training"])

    cm_val = run_split(det, SPLITS["val"], gt)
    cm_test = run_split(det, SPLITS["test"], gt)
    summary["validation"] = {"confusion_matrix": cm_val.tolist(), **metrics_from_cm(cm_val)}
    summary["test"] = {"confusion_matrix": cm_test.tolist(), **metrics_from_cm(cm_test)}
    plot_confusion(cm_val + cm_test, RESULTS / "confusion_matrix.png",
                   "Confusion matrix – validation + test (IoU ≥ 0.5)")
    plot_confusion(cm_test, RESULTS / "confusion_matrix_test.png",
                   "Confusion matrix – test set t1–t5 (IoU ≥ 0.5)")
    print("val:", summary["validation"])
    print("test:", summary["test"])

    make_test_grid(det, RESULTS / "detection_result.jpg")
    make_preprocessing_figure(RESULTS / "preprocessing_comparison.png")

    rows = []
    files = SPLITS["val"] + SPLITS["test"]
    for name, opts in PREPROCESS_VARIANTS.items():
        m = metrics_from_cm(run_split(det, files, gt, **opts))
        rows.append({"preprocessing": name, "accuracy": m["accuracy_all_objects"],
                     "damaged_recall": m["damaged"]["recall"],
                     "damaged_precision": m["damaged"]["precision"],
                     "intact_recall": m["intact"]["recall"],
                     "intact_precision": m["intact"]["precision"]})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(RESULTS / "preprocessing_eval.csv", index=False)
    summary["preprocessing_eval"] = rows

    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("saved ->", RESULTS)


if __name__ == "__main__":
    main()
