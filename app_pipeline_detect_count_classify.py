from pathlib import Path
from collections import Counter
import tempfile
import subprocess
import uuid

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


ROOT = Path("/mnt/d/merged_fruit_36_clean_v2")
YOLO_PATH = ROOT / "yolo_detector" / "best.pt"
CLASSIFIER_PATH = ROOT / "cnn_runs_transfer" / "best_acc_mobilenetv3_36class.pt"
OUT_DIR = ROOT / "streamlit_outputs"
OUT_DIR.mkdir(exist_ok=True)


st.set_page_config(
    page_title="Fruit Detection + Count + Fresh/Rotten",
    layout="wide"
)


@st.cache_resource
def load_yolo():
    if not YOLO_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy YOLO model: {YOLO_PATH}")
    return YOLO(str(YOLO_PATH))


@st.cache_resource
def load_classifier():
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy MobileNetV3 model: {CLASSIFIER_PATH}")

    ckpt = torch.load(CLASSIFIER_PATH, map_location="cpu", weights_only=False)

    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)
    mean = ckpt.get("normalize_mean", [0.485, 0.456, 0.406])
    std = ckpt.get("normalize_std", [0.229, 0.224, 0.225])

    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return model, classes, tf


yolo_model = load_yolo()
classifier, classifier_classes, classifier_tf = load_classifier()


def get_fruit_type_from_class(class_name):
    if "_" in class_name:
        return class_name.rsplit("_", 1)[0]
    return class_name


def classify_crop(crop_img):
    x = classifier_tf(crop_img).unsqueeze(0)

    with torch.no_grad():
        logits = classifier(x)
        probs = torch.softmax(logits, dim=1)[0]

    conf, idx = torch.max(probs, dim=0)

    pred_class = classifier_classes[idx.item()]
    fruit_type = get_fruit_type_from_class(pred_class)

    return fruit_type, pred_class, conf.item()


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def detect_classify_frame(frame_bgr, det_conf, yolo_iou):
    h, w = frame_bgr.shape[:2]

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    result = yolo_model.predict(
        source=pil_img,
        conf=det_conf,
        iou=yolo_iou,
        verbose=False
    )[0]

    detections = []

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        yolo_conf = float(box.conf[0].item())

        crop_rgb = frame_rgb[y1:y2, x1:x2]

        if crop_rgb.size == 0:
            continue

        crop_pil = Image.fromarray(crop_rgb).convert("RGB")

        fruit_type, fresh_rotten_class, cls_conf = classify_crop(crop_pil)

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "fruit_type": fruit_type,
            "fresh_rotten_class": fresh_rotten_class,
            "yolo_confidence": yolo_conf,
            "classifier_confidence": cls_conf,
        })

    return detections


def update_tracks(tracks, detections, frame_idx, next_id, tracking_iou, max_age):
    used_tracks = set()
    draw_rows = []

    for det in detections:
        best_track_id = None
        best_iou = 0.0

        for tid, tr in tracks.items():
            if tid in used_tracks:
                continue

            if frame_idx - tr["last_seen"] > max_age:
                continue

            score = box_iou(det["bbox"], tr["bbox"])

            if score > best_iou:
                best_iou = score
                best_track_id = tid

        if best_track_id is not None and best_iou >= tracking_iou:
            tid = best_track_id
            tr = tracks[tid]
        else:
            tid = next_id
            next_id += 1

            tracks[tid] = {
                "fruit_id": tid,
                "bbox": det["bbox"],
                "first_seen": frame_idx,
                "last_seen": frame_idx,
                "hits": 0,
                "fruit_types": [],
                "fresh_rotten_classes": [],
                "yolo_confidences": [],
                "classifier_confidences": [],
            }

            tr = tracks[tid]

        tr["bbox"] = det["bbox"]
        tr["last_seen"] = frame_idx
        tr["hits"] += 1
        tr["fruit_types"].append(det["fruit_type"])
        tr["fresh_rotten_classes"].append(det["fresh_rotten_class"])
        tr["yolo_confidences"].append(det["yolo_confidence"])
        tr["classifier_confidences"].append(det["classifier_confidence"])

        used_tracks.add(tid)

        draw_rows.append({
            "fruit_id": tid,
            "bbox": det["bbox"],
            "fruit_type": det["fruit_type"],
            "fresh_rotten_class": det["fresh_rotten_class"],
            "yolo_confidence": det["yolo_confidence"],
            "classifier_confidence": det["classifier_confidence"],
        })

    return tracks, next_id, draw_rows


def draw_bbox(frame_bgr, rows):
    out = frame_bgr.copy()

    for row in rows:
        x1, y1, x2, y2 = row["bbox"]

        label = f"ID {row['fruit_id']} | {row['fruit_type']} | {row['fresh_rotten_class']}"

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2

        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        y_text = max(y1 - 8, th + 8)

        cv2.rectangle(
            out,
            (x1, y_text - th - 8),
            (x1 + tw + 8, y_text + 4),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            out,
            label,
            (x1 + 4, y_text - 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return out


def convert_to_h264(input_path):
    output_path = OUT_DIR / f"result_h264_{uuid.uuid4().hex[:8]}.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0 and output_path.exists():
            return output_path

    except Exception:
        pass

    return input_path


def process_video(
    input_path,
    det_conf,
    yolo_iou,
    frame_step,
    max_detect_frames,
    tracking_iou,
    max_age,
    min_hits
):
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError("Không mở được video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_out = OUT_DIR / f"result_raw_{uuid.uuid4().hex[:8]}.mp4"

    writer = cv2.VideoWriter(
        str(raw_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    tracks = {}
    next_id = 1

    frame_idx = 0
    detect_count = 0
    last_draw_rows = []

    progress = st.progress(0)
    status = st.empty()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % frame_step == 0:
            detections = detect_classify_frame(
                frame_bgr=frame,
                det_conf=det_conf,
                yolo_iou=yolo_iou
            )

            tracks, next_id, draw_rows = update_tracks(
                tracks=tracks,
                detections=detections,
                frame_idx=frame_idx,
                next_id=next_id,
                tracking_iou=tracking_iou,
                max_age=max_age
            )

            last_draw_rows = draw_rows
            detect_count += 1

        annotated = draw_bbox(frame, last_draw_rows)
        writer.write(annotated)

        frame_idx += 1

        if total_frames > 0:
            progress.progress(min(frame_idx / total_frames, 1.0))

        status.write(
            f"Đang xử lý frame {frame_idx}/{total_frames} | "
            f"Số lần detect: {detect_count}"
        )

        if detect_count >= max_detect_frames:
            break

    cap.release()
    writer.release()

    progress.progress(1.0)
    status.empty()

    output_path = convert_to_h264(raw_out)

    fruit_rows = []

    for tid, tr in tracks.items():
        if tr["hits"] < min_hits:
            continue

        fruit_type = Counter(tr["fruit_types"]).most_common(1)[0][0]
        fresh_rotten_class = Counter(tr["fresh_rotten_classes"]).most_common(1)[0][0]

        fruit_rows.append({
            "fruit_id": tid,
            "fruit_type": fruit_type,
            "fresh_rotten_class": fresh_rotten_class,
            "observations": tr["hits"],
            "avg_yolo_confidence_percent": round(float(np.mean(tr["yolo_confidences"])) * 100, 2),
            "avg_classifier_confidence_percent": round(float(np.mean(tr["classifier_confidences"])) * 100, 2),
            "first_seen_frame": tr["first_seen"],
            "last_seen_frame": tr["last_seen"],
        })

    fruit_df = pd.DataFrame(fruit_rows)

    return output_path, fruit_df, total_frames, detect_count


def build_summary(fruit_df):
    if fruit_df.empty:
        return pd.DataFrame()

    total = len(fruit_df)
    rows = []

    for fruit_type, group in fruit_df.groupby("fruit_type"):
        count = len(group)
        percent = count / total * 100

        fresh_count = sum(group["fresh_rotten_class"].astype(str).str.endswith("_fresh"))
        rotten_count = sum(group["fresh_rotten_class"].astype(str).str.endswith("_rotten"))
        main_result = Counter(group["fresh_rotten_class"]).most_common(1)[0][0]

        rows.append({
            "fruit_type": fruit_type,
            "total_count": count,
            "percent_of_total": round(percent, 2),
            "fresh_count": int(fresh_count),
            "rotten_count": int(rotten_count),
            "majority_fresh_rotten": main_result,
            "avg_yolo_confidence_percent": round(group["avg_yolo_confidence_percent"].mean(), 2),
            "avg_classifier_confidence_percent": round(group["avg_classifier_confidence_percent"].mean(), 2),
        })

    return pd.DataFrame(rows).sort_values("total_count", ascending=False).reset_index(drop=True)


def build_video_confusion_matrix(fruit_df, true_label):
    if fruit_df.empty:
        return pd.DataFrame()

    y_true = [true_label] * len(fruit_df)
    y_pred = fruit_df["fresh_rotten_class"].astype(str).tolist()

    labels = [true_label]

    for label in y_pred:
        if label not in labels:
            labels.append(label)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"True: {x}" for x in labels],
        columns=[f"Pred: {x}" for x in labels]
    )

    return cm_df


def plot_confusion_matrix(cm_df):
    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(cm_df.values, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(cm_df.columns)),
        yticks=range(len(cm_df.index)),
        xticklabels=cm_df.columns,
        yticklabels=cm_df.index,
        xlabel="Nhãn model dự đoán",
        ylabel="Nhãn thật",
        title="Ma trận nhầm lẫn của video"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            ax.text(j, i, str(cm_df.values[i, j]), ha="center", va="center")

    fig.tight_layout()

    return fig


st.title("🍎 Pipeline nhận dạng + đếm số trái cây")

uploaded_video = st.file_uploader(
    "Upload video",
    type=["mp4", "avi", "mov", "mkv"]
)

true_label = st.selectbox(
    "Nhãn thật của video",
    options=classifier_classes,
    index=0
)

st.markdown("### Tùy chọn xử lý")

col_opt1, col_opt2, col_opt3 = st.columns(3)

with col_opt1:
    detect_option = st.selectbox(
        "Phát hiện YOLO",
        ["Mặc định", "Nhanh", "Cân bằng", "Chính xác"],
        index=0
    )

with col_opt2:
    tracking_option = st.selectbox(
        "Theo dõi & đếm",
        ["Mặc định", "Ổn định", "Nhạy"],
        index=0
    )

with col_opt3:
    scan_option = st.selectbox(
        "Tốc độ quét video",
        ["Mặc định", "Nhanh", "Chuẩn", "Kỹ"],
        index=0
    )


YOLO_PRESETS = {
    "Mặc định": {"det_conf": 0.35, "yolo_iou": 0.45},
    "Nhanh": {"det_conf": 0.40, "yolo_iou": 0.45},
    "Cân bằng": {"det_conf": 0.35, "yolo_iou": 0.45},
    "Chính xác": {"det_conf": 0.25, "yolo_iou": 0.50},
}

TRACKING_PRESETS = {
    "Mặc định": {"tracking_iou": 0.30, "max_age": 25, "min_hits": 3},
    "Ổn định": {"tracking_iou": 0.35, "max_age": 30, "min_hits": 4},
    "Nhạy": {"tracking_iou": 0.25, "max_age": 20, "min_hits": 2},
}

SCAN_PRESETS = {
    "Mặc định": {"frame_step": 1, "max_detect_frames": 1000},
    "Nhanh": {"frame_step": 3, "max_detect_frames": 300},
    "Chuẩn": {"frame_step": 2, "max_detect_frames": 600},
    "Kỹ": {"frame_step": 1, "max_detect_frames": 1000},
}

yolo_cfg = YOLO_PRESETS[detect_option]
tracking_cfg = TRACKING_PRESETS[tracking_option]
scan_cfg = SCAN_PRESETS[scan_option]

det_conf = yolo_cfg["det_conf"]
yolo_iou = yolo_cfg["yolo_iou"]

tracking_iou = tracking_cfg["tracking_iou"]
max_age = tracking_cfg["max_age"]
min_hits = tracking_cfg["min_hits"]

frame_step = scan_cfg["frame_step"]
max_detect_frames = scan_cfg["max_detect_frames"]

with st.expander("Xem cấu hình đang dùng"):
    st.write({
        "YOLO confidence": det_conf,
        "YOLO IOU": yolo_iou,
        "Tracking IOU": tracking_iou,
        "Max age": max_age,
        "Min hits": min_hits,
        "Frame step": frame_step,
        "Max detect frames": max_detect_frames,
    })


if uploaded_video is not None:
    if st.button("Bắt đầu xử lý video", use_container_width=True):
        uploaded_video.seek(0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            input_path = tmp.name

        with st.spinner("Đang chạy YOLO detect + MobileNetV3 classify + tracking..."):
            output_path, fruit_df, total_frames, detect_count = process_video(
                input_path=input_path,
                det_conf=float(det_conf),
                yolo_iou=float(yolo_iou),
                frame_step=int(frame_step),
                max_detect_frames=int(max_detect_frames),
                tracking_iou=float(tracking_iou),
                max_age=int(max_age),
                min_hits=int(min_hits)
            )

        summary_df = build_summary(fruit_df)
        cm_df = build_video_confusion_matrix(fruit_df, true_label)

        left, right = st.columns([1.15, 1])

        with left:
            st.subheader("Video kết quả có bbox")

            with open(output_path, "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes, format="video/mp4")

            st.download_button(
                "Tải video kết quả MP4",
                data=video_bytes,
                file_name="video_result_bbox.mp4",
                mime="video/mp4",
                use_container_width=True
            )

            st.subheader("Ma trận nhầm lẫn")

            if fruit_df.empty:
                st.info("Chưa có fruit_id nào để tạo ma trận nhầm lẫn.")
            else:
                fig_cm = plot_confusion_matrix(cm_df)
                st.pyplot(fig_cm)

                with st.expander("Xem ma trận dạng bảng"):
                    st.dataframe(cm_df, use_container_width=True)

                st.download_button(
                    "Tải ma trận nhầm lẫn CSV",
                    data=cm_df.to_csv(encoding="utf-8-sig").encode("utf-8-sig"),
                    file_name="video_confusion_matrix.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with right:
            st.subheader("Kết quả nhận diện")

            st.success(f"Đã xử lý. Số lần detect: {detect_count} / tổng frame: {total_frames}")
            st.metric("Tổng số trái cây ước lượng", len(fruit_df))

            if fruit_df.empty:
                st.warning("Không phát hiện được trái cây.")
            else:
                top = summary_df.iloc[0]

                st.metric(
                    "Loại chiếm nhiều nhất",
                    top["fruit_type"],
                    f"{top['percent_of_total']:.2f}%"
                )

                st.subheader("Bảng tổng hợp")
                st.dataframe(summary_df, use_container_width=True)

                st.subheader("Phần trăm theo loại trái")
                st.bar_chart(summary_df.set_index("fruit_type")["percent_of_total"])

                st.subheader("Chi tiết từng trái")
                st.dataframe(fruit_df, use_container_width=True)

                st.download_button(
                    "Tải bảng tổng hợp CSV",
                    data=summary_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="fruit_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.download_button(
                    "Tải chi tiết từng trái CSV",
                    data=fruit_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="fruit_detail.csv",
                    mime="text/csv",
                    use_container_width=True
                )
