"""
detection.py
------------
โหลดโมเดล YOLOv7 (best.pt) และตรวจจับฝาขวดปกติ (intact) / ฝาขวดมีตำหนิ (damaged)
บนภาพหรือวิดีโอ พร้อมวาด Bounding Box และคำนวณสถิติ

ใช้งานแบบ command line:
    python src/detection.py --source images/input.jpg --output images/result.jpg
    python src/detection.py --source demo/input.mp4 --output demo/result.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]          # โฟลเดอร์โปรเจกต์
YOLOV7_DIR = ROOT / "yolov7"                        # โค้ด YOLOv7 (ต้องมีเพื่อโหลดโครงสร้างโมเดล)
DEFAULT_WEIGHTS = ROOT / "best.pt"

if str(YOLOV7_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV7_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import enhance, to_tensor_input  # noqa: E402

# สีของกรอบ (BGR): แดง = มีตำหนิ, เขียว = ปกติ
COLORS = {"damaged": (40, 40, 230), "intact": (60, 180, 60)}


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    label: str

    def as_dict(self):
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
            "width": self.x2 - self.x1, "height": self.y2 - self.y1,
        }


@dataclass
class DetectionResult:
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0

    @property
    def n_damaged(self) -> int:
        return sum(d.label == "damaged" for d in self.detections)

    @property
    def n_intact(self) -> int:
        return sum(d.label == "intact" for d in self.detections)

    @property
    def total(self) -> int:
        return len(self.detections)

    @property
    def defect_rate(self) -> float:
        """สัดส่วนฝาที่มีตำหนิ (%)"""
        return 100.0 * self.n_damaged / self.total if self.total else 0.0

    def summary(self) -> dict:
        confs = [d.confidence for d in self.detections]
        return {
            "total": self.total,
            "intact": self.n_intact,
            "damaged": self.n_damaged,
            "defect_rate_%": round(self.defect_rate, 2),
            "mean_confidence": round(float(np.mean(confs)), 3) if confs else 0.0,
            "inference_ms": round(self.inference_ms, 1),
            "status": "FAIL (พบฝามีตำหนิ)" if self.n_damaged else "PASS (ไม่พบตำหนิ)",
        }


def _load_checkpoint(weights: str | Path, device: torch.device) -> nn.Module:
    """โหลด YOLOv7 checkpoint (ปิด weights_only เพราะไฟล์ .pt ของ YOLOv7 เก็บทั้ง object โมเดล)"""
    if not YOLOV7_DIR.exists():
        raise FileNotFoundError(
            f"ไม่พบโฟลเดอร์ yolov7 ที่ {YOLOV7_DIR}\n"
            "ให้รัน: git clone https://github.com/WongKinYiu/yolov7"
        )
    import models.common as common  # noqa: F401  (ต้อง import เพื่อให้ unpickle ได้)
    from models.yolo import Model  # noqa: F401

    ckpt = torch.load(str(weights), map_location=device, weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float().fuse().eval()

    # ความเข้ากันได้กับ PyTorch รุ่นใหม่ (เหมือน attempt_load ของ YOLOv7)
    for m in model.modules():
        if type(m) in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU):
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None
        elif type(m) is common.Conv:
            m._non_persistent_buffers_set = set()
    return model.to(device)


class DefectDetector:
    """ตัวตรวจจับตำหนิฝาขวดด้วย YOLOv7"""

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        device: Optional[str] = None,
        img_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        agnostic_nms: bool = True,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = _load_checkpoint(weights, self.device)
        self.half = self.device.type == "cuda"
        if self.half:
            self.model.half()
        self.stride = int(self.model.stride.max())
        self.img_size = int(np.ceil(img_size / self.stride) * self.stride)
        self.names = list(self.model.names)  # ['damaged', 'intact']
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        # agnostic NMS: ฝา 1 ใบได้ผลลัพธ์เดียว (ไม่ซ้อน intact + damaged ที่ตำแหน่งเดียวกัน)
        self.agnostic_nms = agnostic_nms

        # warm-up
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.img_size, self.img_size, device=self.device)
            self.model(dummy.half() if self.half else dummy)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def detect(self, img_bgr: np.ndarray, **enhance_opts) -> DetectionResult:
        """ตรวจจับวัตถุในภาพ BGR หนึ่งภาพ"""
        from utils.general import non_max_suppression, scale_coords

        src = enhance(img_bgr, **enhance_opts) if enhance_opts else img_bgr
        x, lb_shape = to_tensor_input(src, self.img_size, self.stride)
        x = torch.from_numpy(x).to(self.device)
        if self.half:
            x = x.half()

        t0 = time.perf_counter()
        pred = self.model(x)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=self.agnostic_nms)[0]
        ms = (time.perf_counter() - t0) * 1000

        dets: list[Detection] = []
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(lb_shape, pred[:, :4], img_bgr.shape).round()
            for *xyxy, conf, cls in pred.cpu().numpy():
                c = int(cls)
                dets.append(Detection(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]),
                                      float(conf), c, self.names[c]))
        dets.sort(key=lambda d: (d.y1 // 80, d.x1))  # เรียงซ้าย->ขวา บน->ล่าง
        return DetectionResult(dets, ms)

    # ------------------------------------------------------------------
    @staticmethod
    def draw(img_bgr: np.ndarray, result: DetectionResult, show_summary: bool = True) -> np.ndarray:
        """วาด Bounding Box + label + ความมั่นใจ ลงบนภาพ (คืนภาพใหม่)"""
        out = img_bgr.copy()
        h, w = out.shape[:2]
        lw = max(2, round((h + w) / 600))
        fs = max(0.45, lw / 4)
        for i, d in enumerate(result.detections, 1):
            color = COLORS.get(d.label, (255, 160, 0))
            cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, lw, cv2.LINE_AA)
            text = f"#{i} {d.label} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ty = d.y1 - 4 if d.y1 - th - 6 > 0 else d.y1 + th + 6
            cv2.rectangle(out, (d.x1, ty - th - 4), (d.x1 + tw + 4, ty + 3), color, -1)
            cv2.putText(out, text, (d.x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), 1, cv2.LINE_AA)

        if show_summary:
            txt = f"Intact: {result.n_intact}  Damaged: {result.n_damaged}  Defect: {result.defect_rate:.1f}%"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs + 0.1, 2)
            cv2.rectangle(out, (0, h - th - 14), (tw + 16, h), (30, 30, 30), -1)
            cv2.putText(out, txt, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, fs + 0.1,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    def detect_video(
        self,
        src_path: str,
        dst_path: str,
        frame_step: int = 1,
        max_frames: Optional[int] = None,
        progress: Optional[Callable[[float], None]] = None,
        **enhance_opts,
    ) -> dict:
        """
        ตรวจจับทุก ๆ `frame_step` เฟรมของวิดีโอ แล้วบันทึกวิดีโอผลลัพธ์ (H.264 ถ้ามี ffmpeg)
        คืนค่าสถิติรวมของวิดีโอ
        """
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            raise ValueError(f"เปิดวิดีโอไม่ได้: {src_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if max_frames:
            n_total = min(n_total, max_frames) if n_total else max_frames

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        per_frame, last, idx = [], None, 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and idx >= max_frames):
                break
            if idx % frame_step == 0 or last is None:
                last = self.detect(frame, **enhance_opts)
                per_frame.append({"frame": idx, "time_s": round(idx / fps, 2),
                                  "intact": last.n_intact, "damaged": last.n_damaged,
                                  "total": last.total, "inference_ms": round(last.inference_ms, 1)})
            writer.write(self.draw(frame, last))
            idx += 1
            if progress and n_total:
                progress(min(idx / n_total, 1.0))
        cap.release()
        writer.release()

        _to_browser_mp4(tmp, dst_path)

        if not per_frame:
            return {"frames": 0}
        dmg = np.array([p["damaged"] for p in per_frame])
        ok_ = np.array([p["intact"] for p in per_frame])
        return {
            "frames": idx,
            "frames_analysed": len(per_frame),
            "fps": round(fps, 2),
            "max_damaged_in_frame": int(dmg.max()),
            "max_intact_in_frame": int(ok_.max()),
            "avg_damaged_per_frame": round(float(dmg.mean()), 2),
            "avg_intact_per_frame": round(float(ok_.mean()), 2),
            "frames_with_defect_%": round(100 * float((dmg > 0).mean()), 1),
            "avg_inference_ms": round(float(np.mean([p["inference_ms"] for p in per_frame])), 1),
            "per_frame": per_frame,
        }


def _ffmpeg_exe() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _to_browser_mp4(src: str, dst: str):
    """แปลงวิดีโอเป็น H.264 เพื่อให้เล่นบนเว็บเบราว์เซอร์ได้ (ถ้าไม่มี ffmpeg จะคัดลอกไฟล์เดิม)"""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    exe = _ffmpeg_exe()
    if exe:
        r = subprocess.run([exe, "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264",
                            "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst])
        if r.returncode == 0:
            os.remove(src)
            return
    shutil.move(src, dst)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Jar lid defect detection (YOLOv7)")
    ap.add_argument("--source", required=True, help="ไฟล์ภาพ / วิดีโอ / โฟลเดอร์ภาพ")
    ap.add_argument("--output", default="results/detection_result.jpg")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--clahe", action="store_true", help="ใช้ CLAHE ปรับคอนทราสต์ก่อนตรวจจับ")
    args = ap.parse_args()

    det = DefectDetector(args.weights, conf_thres=args.conf, iou_thres=args.iou)
    opts = {"use_clahe": True} if args.clahe else {}
    src = Path(args.source)

    if src.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        stats = det.detect_video(str(src), args.output, **opts)
        stats.pop("per_frame", None)
        print(stats)
        return

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
        if src.is_dir() else [src]
    out = Path(args.output)
    for f in files:
        img = cv2.imread(str(f))
        res = det.detect(img, **opts)
        target = out / f.name if src.is_dir() else out
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), det.draw(img, res))
        print(f.name, res.summary())


if __name__ == "__main__":
    main()
