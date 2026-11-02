"""
app.py — Web Application ตรวจจับตำหนิฝาขวด (Jar Lid Defect Detection) ด้วย YOLOv7 + Streamlit

รัน:  streamlit run app.py
"""

import json
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src.detection import DefectDetector
from src.preprocessing import bgr_to_rgb, enhance, read_image_bytes

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SAMPLES = ROOT / "images" / "test"

st.set_page_config(page_title="Jar Lid Defect Detection", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="กำลังโหลดโมเดล YOLOv7 ...")
def load_detector() -> DefectDetector:
    return DefectDetector(ROOT / "best.pt")


# ---------------------------------------------------------------------------
# Sidebar : การตั้งค่า
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ การตั้งค่า")
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                         help="ความมั่นใจขั้นต่ำของกรอบที่จะแสดง")
iou = st.sidebar.slider("IoU threshold (NMS)", 0.1, 0.9, 0.45, 0.05,
                        help="ค่าการซ้อนทับสำหรับรวมกรอบที่ซ้ำกัน")

st.sidebar.subheader("🖼️ Image Preprocessing (OpenCV)")
opts = {
    "use_clahe": st.sidebar.checkbox("CLAHE (ปรับคอนทราสต์เฉพาะส่วน)"),
    "use_glare": st.sidebar.checkbox("ลดแสงสะท้อน (Glare reduction)"),
    "use_denoise": st.sidebar.checkbox("ลด Noise (Gaussian blur)"),
    "use_sharpen": st.sidebar.checkbox("เพิ่มความคมชัด (Sharpen)"),
    "brightness": st.sidebar.slider("Brightness", -60, 60, 0, 5),
    "contrast": st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.1),
}
use_pre = any(v for k, v in opts.items() if k not in ("brightness", "contrast")) \
    or opts["brightness"] != 0 or opts["contrast"] != 1.0
pre_opts = opts if use_pre else {}

detector = load_detector()
detector.conf_thres, detector.iou_thres = conf, iou

st.sidebar.caption(f"Device: `{detector.device}` · Classes: {', '.join(detector.names)}")

# ---------------------------------------------------------------------------
st.title("🔍 Jar Lid Defect Detection")
st.caption("ตรวจจับฝาขวดปกติ (intact) และฝาขวดที่มีตำหนิ (damaged) ด้วย YOLOv7 · Python + OpenCV + Streamlit")

tab_img, tab_vid, tab_eval, tab_about = st.tabs(
    ["📷 ตรวจจับจากภาพ", "🎞️ ตรวจจับจากวิดีโอ", "📊 ผลการประเมินโมเดล", "ℹ️ เกี่ยวกับโปรเจกต์"])

# ---------------------------------------------------------------------------
# 1) Image
# ---------------------------------------------------------------------------
with tab_img:
    c1, c2 = st.columns([2, 1])
    uploads = c1.file_uploader("อัปโหลดภาพ (เลือกได้หลายไฟล์)", type=["jpg", "jpeg", "png", "bmp"],
                               accept_multiple_files=True)
    samples = sorted(SAMPLES.glob("*.JPG")) + sorted(SAMPLES.glob("*.jpg"))
    sample = c2.selectbox("หรือเลือกภาพตัวอย่าง (ชุด Test)", ["—"] + [p.name for p in samples])

    items = [(u.name, read_image_bytes(u.getvalue())) for u in uploads or []]
    if not items and sample != "—":
        items = [(sample, cv2.imread(str(SAMPLES / sample)))]

    if not items:
        st.info("อัปโหลดภาพหรือเลือกภาพตัวอย่างเพื่อเริ่มตรวจจับ")
    batch_rows = []
    for name, img in items:
        res = detector.detect(img, **pre_opts)
        drawn = detector.draw(img, res)
        s = res.summary()
        batch_rows.append({"file": name, **s})

        st.markdown(f"### {name}")
        cols = st.columns(3 if use_pre else 2)
        cols[0].image(bgr_to_rgb(img), caption="ภาพต้นฉบับ", width="stretch")
        if use_pre:
            cols[1].image(bgr_to_rgb(enhance(img, **pre_opts)), caption="หลัง Preprocessing",
                          width="stretch")
        cols[-1].image(bgr_to_rgb(drawn), caption="ผลการตรวจจับ", width="stretch")

        m = st.columns(5)
        m[0].metric("ฝาทั้งหมด", s["total"])
        m[1].metric("✅ ปกติ (intact)", s["intact"])
        m[2].metric("❌ มีตำหนิ (damaged)", s["damaged"])
        m[3].metric("อัตราตำหนิ", f"{s['defect_rate_%']:.1f}%")
        m[4].metric("เวลาประมวลผล", f"{s['inference_ms']:.0f} ms")
        (st.error if res.n_damaged else st.success)(f"สถานะ: {s['status']}")

        with st.expander("รายละเอียดแต่ละกรอบ (Bounding Box)"):
            if res.detections:
                df = pd.DataFrame([{"#": i, **d.as_dict()} for i, d in enumerate(res.detections, 1)])
                st.dataframe(df, hide_index=True, width="stretch")
                st.bar_chart(df.groupby("label")["confidence"].mean(), y_label="mean confidence")
            else:
                st.write("ไม่พบวัตถุ — ลองลด Confidence threshold")

        ok, buf = cv2.imencode(".jpg", drawn)
        st.download_button("⬇️ ดาวน์โหลดภาพผลลัพธ์", buf.tobytes(), file_name=f"result_{Path(name).stem}.jpg",
                           mime="image/jpeg", key=f"dl_{name}")
        st.divider()

    if len(batch_rows) > 1:
        st.subheader("📋 สรุปผลทุกภาพ")
        bdf = pd.DataFrame(batch_rows)
        st.dataframe(bdf, hide_index=True, width="stretch")
        tot_i, tot_d = int(bdf["intact"].sum()), int(bdf["damaged"].sum())
        st.write(f"รวม **{tot_i + tot_d}** ฝา · ปกติ **{tot_i}** · มีตำหนิ **{tot_d}** "
                 f"· อัตราตำหนิรวม **{100 * tot_d / max(tot_i + tot_d, 1):.1f}%**")
        st.bar_chart(bdf.set_index("file")[["damaged", "intact"]], color=["#e03131", "#2f9e44"])
        st.download_button("⬇️ ดาวน์โหลดสรุป (CSV)", bdf.to_csv(index=False).encode("utf-8-sig"),
                           "detection_summary.csv", "text/csv")

# ---------------------------------------------------------------------------
# 2) Video
# ---------------------------------------------------------------------------
with tab_vid:
    vfile = st.file_uploader("อัปโหลดวิดีโอ", type=["mp4", "avi", "mov", "mkv"])
    c1, c2 = st.columns(2)
    step = c1.number_input("ตรวจจับทุก ๆ N เฟรม (ค่ามาก = เร็วขึ้น)", 1, 30, 5,
                           help="เฟรมที่ไม่ได้ตรวจจับจะใช้ผลของเฟรมล่าสุด")
    max_sec = c2.number_input("ประมวลผลสูงสุด (วินาที)", 1, 300, 20)

    if vfile and st.button("▶️ เริ่มตรวจจับวิดีโอ", type="primary"):
        src = tempfile.NamedTemporaryFile(suffix=Path(vfile.name).suffix, delete=False)
        src.write(vfile.getvalue())
        src.close()
        fps = cv2.VideoCapture(src.name).get(cv2.CAP_PROP_FPS) or 25
        out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        bar = st.progress(0.0, text="กำลังประมวลผล ...")
        stats = detector.detect_video(src.name, out, frame_step=int(step),
                                      max_frames=int(max_sec * fps),
                                      progress=lambda p: bar.progress(p, text=f"กำลังประมวลผล {p:.0%}"),
                                      **pre_opts)
        bar.empty()
        st.session_state["video"] = (out, stats)

    if "video" in st.session_state:
        out, stats = st.session_state["video"]
        st.video(out)
        m = st.columns(4)
        m[0].metric("เฟรมทั้งหมด", stats["frames"])
        m[1].metric("ฝามีตำหนิสูงสุด/เฟรม", stats["max_damaged_in_frame"])
        m[2].metric("ฝาปกติสูงสุด/เฟรม", stats["max_intact_in_frame"])
        m[3].metric("เฟรมที่พบตำหนิ", f"{stats['frames_with_defect_%']}%")
        pf = pd.DataFrame(stats["per_frame"])
        st.line_chart(pf.set_index("time_s")[["damaged", "intact"]], color=["#e03131", "#2f9e44"],
                      x_label="เวลา (วินาที)", y_label="จำนวนฝา")
        st.download_button("⬇️ ดาวน์โหลดวิดีโอผลลัพธ์", open(out, "rb").read(), "result.mp4", "video/mp4")

# ---------------------------------------------------------------------------
# 3) Evaluation
# ---------------------------------------------------------------------------
with tab_eval:
    mfile = RESULTS / "metrics.json"
    if not mfile.exists():
        st.warning("ยังไม่มีผลการประเมิน — รัน `python src/evaluate.py` ก่อน")
    else:
        mt = json.loads(mfile.read_text(encoding="utf-8"))
        tr = mt.get("training", {})
        st.subheader("ผลการฝึกโมเดล (YOLOv7, Validation set)")
        m = st.columns(4)
        m[0].metric("Best epoch", tr.get("best_epoch", "-"))
        m[1].metric("mAP@0.5", tr.get("best_val_mAP50", "-"))
        m[2].metric("mAP@0.5:0.95", tr.get("best_val_mAP50_95", "-"))
        m[3].metric("Precision / Recall", f"{tr.get('last_precision', '-')} / {tr.get('last_recall', '-')}")
        if (RESULTS / "results.png").exists():
            st.image(str(RESULTS / "results.png"), caption="Learning curves (results.png)")

        st.subheader("ผลบนชุดข้อมูล Validation และ Test (IoU ≥ 0.5)")
        rows = []
        for split in ("validation", "test"):
            for c in ("damaged", "intact"):
                rows.append({"split": split, "class": c, **mt[split][c]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.write(f"ความถูกต้อง (ทุกวัตถุ): Validation **{mt['validation']['accuracy_all_objects']:.1%}** · "
                 f"Test **{mt['test']['accuracy_all_objects']:.1%}**")
        c1, c2 = st.columns(2)
        c1.image(str(RESULTS / "confusion_matrix.png"), caption="Confusion matrix (Validation + Test)")
        c2.image(str(RESULTS / "confusion_matrix_test.png"), caption="Confusion matrix (Test t1–t5)")
        st.image(str(RESULTS / "detection_result.jpg"), caption="ผลตรวจจับชุด Test")

        st.subheader("การทดลอง Image Processing")
        st.image(str(RESULTS / "preprocessing_comparison.png"))
        st.dataframe(pd.read_csv(RESULTS / "preprocessing_eval.csv"), hide_index=True,
                     width="stretch")
        st.caption("ทดสอบบน Validation + Test (44 ภาพ) เพื่อดูผลของการปรับภาพแต่ละแบบต่อความแม่นยำของโมเดล")

# ---------------------------------------------------------------------------
# 4) About
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("""
**Product Defect Detection – Jar Lids**

- **Dataset:** Jar Lids (ภาพ 640×480, annotate ด้วย VGG Image Annotator) — Train 124 ภาพ, Validation 39 ภาพ, Test 5 ภาพ
- **Classes:** `damaged` (ฝามีตำหนิ: รู, บุบ, รอยขีดข่วน) และ `intact` (ฝาปกติ)
- **Model:** YOLOv7 fine-tune จาก `yolov7.pt` (img 640, batch 16, ปิด mosaic/scale, flipud 0.5)
- **Pipeline:** รับภาพ/วิดีโอ → OpenCV preprocessing (ทางเลือก) → YOLOv7 → NMS → วาด Bounding Box → นับจำนวน/คำนวณอัตราตำหนิ
""")
