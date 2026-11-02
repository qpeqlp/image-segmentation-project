"""
preprocessing.py
----------------
ฟังก์ชันประมวลผลภาพด้วย OpenCV สำหรับระบบตรวจจับตำหนิฝาขวด (Jar Lid Defect Detection)

ประกอบด้วย
- การอ่านภาพจากไฟล์ / bytes (ที่อัปโหลดผ่าน Streamlit)
- การปรับภาพก่อนตรวจจับ (Image Enhancement) : ปรับความสว่าง/คอนทราสต์, CLAHE,
  ลด Noise, เพิ่มความคมชัด, ลดแสงสะท้อน
- Letterbox : ย่อ/ขยายภาพให้เป็นขนาดที่โมเดล YOLOv7 ต้องการ (640x640) โดยคงอัตราส่วนภาพ
- การแปลงภาพเป็น Tensor สำหรับป้อนเข้าโมเดล
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# อ่านภาพ
# ---------------------------------------------------------------------------
def read_image(path: str) -> np.ndarray:
    """อ่านภาพจาก path คืนค่าเป็นภาพ BGR (รูปแบบของ OpenCV)"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"อ่านภาพไม่ได้: {path}")
    return img


def read_image_bytes(data: bytes) -> np.ndarray:
    """แปลง bytes ของไฟล์ภาพ (เช่นไฟล์ที่อัปโหลดผ่านเว็บ) เป็นภาพ BGR"""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("ไฟล์ที่อัปโหลดไม่ใช่ภาพที่ OpenCV อ่านได้")
    return img


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Image Enhancement (ทางเลือก ก่อนส่งเข้าโมเดล)
# ---------------------------------------------------------------------------
def adjust_brightness_contrast(img: np.ndarray, brightness: int = 0, contrast: float = 1.0) -> np.ndarray:
    """ปรับความสว่าง (beta) และคอนทราสต์ (alpha): out = alpha*img + beta"""
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) บนช่อง L ของ LAB
    ช่วยให้ภาพที่แสงไม่สม่ำเสมอเห็นรอยบุบ/รอยขีดข่วนบนฝาชัดขึ้น โดยไม่ทำให้สีเพี้ยน
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """ลด Noise ด้วย Gaussian Blur (ksize ต้องเป็นเลขคี่)"""
    ksize = max(1, ksize | 1)
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def sharpen(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """เพิ่มความคมชัดด้วยเทคนิค Unsharp Mask"""
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def reduce_glare(img: np.ndarray, threshold: int = 240) -> np.ndarray:
    """
    ลดแสงสะท้อนบนฝาโลหะ: หา pixel ที่สว่างจัด (เกือบขาว) แล้วเติมด้วย Inpainting
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def enhance(
    img: np.ndarray,
    use_clahe: bool = False,
    use_denoise: bool = False,
    use_sharpen: bool = False,
    use_glare: bool = False,
    brightness: int = 0,
    contrast: float = 1.0,
) -> np.ndarray:
    """รวมขั้นตอนปรับภาพทั้งหมดตามตัวเลือกที่ผู้ใช้เลือก (ค่าเริ่มต้น = ไม่ปรับภาพ)"""
    out = img.copy()
    if brightness != 0 or contrast != 1.0:
        out = adjust_brightness_contrast(out, brightness, contrast)
    if use_glare:
        out = reduce_glare(out)
    if use_denoise:
        out = denoise(out)
    if use_clahe:
        out = apply_clahe(out)
    if use_sharpen:
        out = sharpen(out)
    return out


# ---------------------------------------------------------------------------
# เตรียมภาพเข้าโมเดล YOLOv7
# ---------------------------------------------------------------------------
def letterbox(img: np.ndarray, new_shape: int = 640, color=(114, 114, 114), stride: int = 32):
    """
    ย่อ/ขยายภาพโดยคงอัตราส่วน แล้วเติมขอบ (padding) ให้ได้ขนาดหารด้วย stride ลงตัว
    คืนค่า (ภาพใหม่, อัตราส่วนการย่อ, (pad_w, pad_h))
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = np.mod(dw, stride) / 2, np.mod(dh, stride) / 2  # minimum rectangle

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def to_tensor_input(img_bgr: np.ndarray, img_size: int = 640, stride: int = 32):
    """
    แปลงภาพ BGR เป็น array รูปแบบ (1, 3, H, W) ค่า 0-1 ที่ YOLOv7 ต้องการ
    คืนค่า (array, ขนาดภาพหลัง letterbox)
    """
    lb, _, _ = letterbox(img_bgr, img_size, stride=stride)
    x = lb[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return x[None], lb.shape[:2]
