"""
app.py — Gradio web application for AI-based industrial surface defect inspection.

Tabs:
  1. Single Inspection   — one image -> boxes, defect type, confidence, PASS/FAIL
  2. Robustness Test     — perturb an image (brightness/rotation/blur/noise) then inspect
  3. Batch Inspection    — many images -> table + downloadable CSV

Run locally:  python app.py
"""

import os
import cv2
import gradio as gr

from utils.detection import inspect, model_info, DEFAULT_CONF
from utils.robustness import apply_transforms, describe
from utils.report import make_row, build_report, save_csv, summary_line

TITLE = "AI Industrial Surface Defect Inspection"
DESCRIPTION = (
    "Upload an image of a steel surface. A YOLO object-detection model (CNN) locates and "
    "classifies surface defects, then the system issues an automated PASS/FAIL result.\n\n"
    "Dataset: NEU-DET (6 hot-rolled steel defect classes) · Model: Ultralytics YOLO · "
    "Built for a 2-day hackathon — see README for limitations."
)


# ---------------------------------------------------------------------------
# Helpers shared by the tabs
# ---------------------------------------------------------------------------

def status_markdown(status: str) -> str:
    if status == "PASS":
        return "## ✅ PASS — no defects detected"
    if status == "FAIL":
        return "## ❌ FAIL — defects detected"
    return f"## ⚠️ {status}"


def format_detections(detections: list) -> str:
    if not detections:
        return "_No detections above threshold._"
    lines = ["| # | Defect type | Confidence | Bounding box (x1, y1, x2, y2) |", "|---|---|---|---|"]
    for i, d in enumerate(detections, 1):
        lines.append(f"| {i} | {d['class']} | {d['confidence']:.2f} | {d['bbox']} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 1 — Single inspection
# ---------------------------------------------------------------------------

def single_inspection(image, conf):
    if image is None:
        return None, status_markdown("Upload an image first"), "", ""
    try:
        r = inspect(image, conf)
    except Exception as e:  # noqa: BLE001
        return None, status_markdown(f"ERROR: {e}"), "", ""

    summary = (
        f"**Defect count:** {r['defect_count']}  \n"
        f"**Primary defect:** {r['primary_class']}  \n"
        f"**Highest confidence:** {r['max_confidence']:.2f}  \n"
        f"**Threshold used:** {conf:.2f} (demo threshold)"
    )
    return r["annotated"], status_markdown(r["status"]), summary + "\n\n" + r["explanation"], format_detections(r["detections"])


# ---------------------------------------------------------------------------
# Tab 2 — Robustness test
# ---------------------------------------------------------------------------

def robustness_test(image, brightness, angle, blur_strength, noise_sigma, conf):
    if image is None:
        return None, None, status_markdown("Upload an image first"), ""
    try:
        transformed = apply_transforms(image, brightness, angle, blur_strength, noise_sigma)
        r = inspect(transformed, conf)
    except Exception as e:  # noqa: BLE001
        return None, None, status_markdown(f"ERROR: {e}"), ""

    text = (
        f"**Applied:** {describe(brightness, angle, blur_strength, noise_sigma)}  \n"
        f"**Defect count:** {r['defect_count']} · **Primary defect:** {r['primary_class']} · "
        f"**Max confidence:** {r['max_confidence']:.2f}\n\n"
        "_This tab demonstrates behaviour under controlled perturbations. It is not proof that "
        "the model works under every real-world condition._\n\n"
        + format_detections(r["detections"])
    )
    return transformed, r["annotated"], status_markdown(r["status"]), text


# ---------------------------------------------------------------------------
# Tab 3 — Batch inspection
# ---------------------------------------------------------------------------

def batch_inspection(files, conf):
    if not files:
        return None, None, "Upload one or more images."
    rows = []
    for f in files:
        path = f.name if hasattr(f, "name") else str(f)
        filename = os.path.basename(path)
        bgr = cv2.imread(path)
        if bgr is None:
            rows.append({"filename": filename, "status": "UNREADABLE", "defect_count": 0,
                         "primary_defect": "-", "max_confidence": 0.0, "all_defects": "-"})
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            r = inspect(rgb, conf)
            rows.append(make_row(filename, r))
        except Exception as e:  # noqa: BLE001
            rows.append({"filename": filename, "status": f"ERROR: {e}", "defect_count": 0,
                         "primary_defect": "-", "max_confidence": 0.0, "all_defects": "-"})
    df = build_report(rows)
    csv_path = save_csv(df)
    return df, csv_path, summary_line(df)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"#  {TITLE}")
    gr.Markdown(DESCRIPTION)
    gr.Markdown(f"**Model info:** {model_info()}")

    # ---- Tab 1 -----------------------------------------------------------
    with gr.Tab("1 · Single Inspection"):
        with gr.Row():
            with gr.Column():
                in_img = gr.Image(type="numpy", label="Upload surface image")
                in_conf = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05,
                                    label="Confidence threshold (demo default 0.25)")
                run_btn = gr.Button("Run Inspection", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Annotated result")
                out_status = gr.Markdown(status_markdown("Awaiting image"))
                out_summary = gr.Markdown()
                out_table = gr.Markdown()
        run_btn.click(single_inspection, [in_img, in_conf], [out_img, out_status, out_summary, out_table])

    # ---- Tab 2 -----------------------------------------------------------
    with gr.Tab("2 · Robustness Test"):
        gr.Markdown("Apply controlled variations, then inspect the transformed image.")
        with gr.Row():
            with gr.Column():
                rb_img = gr.Image(type="numpy", label="Upload surface image")
                rb_bright = gr.Slider(0.3, 1.8, value=1.0, step=0.05, label="Brightness factor (1.0 = original)")
                rb_angle = gr.Slider(-180, 180, value=0, step=5, label="Rotation (degrees)")
                rb_blur = gr.Slider(0, 8, value=0, step=1, label="Blur strength (0 = off)")
                rb_noise = gr.Slider(0, 60, value=0, step=5, label="Noise sigma (0 = off)")
                rb_conf = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05, label="Confidence threshold")
                rb_btn = gr.Button("Transform & Inspect", variant="primary")
            with gr.Column():
                rb_out_t = gr.Image(label="Transformed image")
                rb_out_a = gr.Image(label="Detection on transformed image")
                rb_status = gr.Markdown(status_markdown("Awaiting image"))
                rb_text = gr.Markdown()
        rb_btn.click(robustness_test,
                     [rb_img, rb_bright, rb_angle, rb_blur, rb_noise, rb_conf],
                     [rb_out_t, rb_out_a, rb_status, rb_text])

    # ---- Tab 3 -----------------------------------------------------------
    with gr.Tab("3 · Batch Inspection"):
        gr.Markdown("Upload several images. Each is inspected and the results are exported to CSV.")
        bt_files = gr.File(file_count="multiple", file_types=["image"], label="Upload images")
        bt_conf = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05, label="Confidence threshold")
        bt_btn = gr.Button("Run Batch Inspection", variant="primary")
        bt_summary = gr.Markdown()
        bt_table = gr.Dataframe(label="Inspection report", wrap=True)
        bt_csv = gr.File(label="Download CSV report")
        bt_btn.click(batch_inspection, [bt_files, bt_conf], [bt_table, bt_csv, bt_summary])

    
if __name__ == "__main__":
    demo.launch()