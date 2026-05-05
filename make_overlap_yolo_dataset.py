import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_image_label_pairs(src_root):
    src_root = Path(src_root)

    pairs = []

    label_files = list(src_root.rglob("*.txt"))

    for label_path in label_files:
        if "labels" not in label_path.parts:
            continue

        rel = label_path.relative_to(src_root)

        rel_parts = list(rel.parts)
        rel_parts[rel_parts.index("labels")] = "images"

        image_rel_base = Path(*rel_parts).with_suffix("")

        image_path = None

        for ext in IMG_EXTS:
            candidate = src_root / image_rel_base.with_suffix(ext)
            if candidate.exists():
                image_path = candidate
                break

        if image_path is not None:
            pairs.append((image_path, label_path))

    return pairs


def read_yolo_labels(label_path):
    labels = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            labels.append((cls_id, xc, yc, w, h))

    return labels


def yolo_to_xyxy(label, img_w, img_h):
    cls_id, xc, yc, w, h = label

    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))

    return cls_id, x1, y1, x2, y2


def xyxy_to_yolo(cls_id, x1, y1, x2, y2, img_w, img_h):
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 2 or bh <= 2:
        return None

    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    w = bw / img_w
    h = bh / img_h

    return f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def make_object_mask(crop):
    """
    Tạo mask đơn giản bằng GrabCut.
    Mục tiêu: tách trái cây ra khỏi nền tương đối tốt để dán chồng chéo.
    """
    h, w = crop.shape[:2]

    if h < 10 or w < 10:
        return np.ones((h, w), dtype=np.uint8) * 255

    mask = np.zeros((h, w), np.uint8)

    rect_margin_x = max(2, int(w * 0.08))
    rect_margin_y = max(2, int(h * 0.08))

    rect = (
        rect_margin_x,
        rect_margin_y,
        max(1, w - 2 * rect_margin_x),
        max(1, h - 2 * rect_margin_y),
    )

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(crop, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype("uint8")

        kernel = np.ones((3, 3), np.uint8)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

        if mask2.sum() < 0.1 * 255 * h * w:
            return np.ones((h, w), dtype=np.uint8) * 255

        return mask2

    except Exception:
        return np.ones((h, w), dtype=np.uint8) * 255


def extract_objects(pairs, max_objects=None):
    objects = []

    for image_path, label_path in pairs:
        img = cv2.imread(str(image_path))

        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        labels = read_yolo_labels(label_path)

        for label in labels:
            cls_id, x1, y1, x2, y2 = yolo_to_xyxy(label, img_w, img_h)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2].copy()

            if crop.shape[0] < 20 or crop.shape[1] < 20:
                continue

            mask = make_object_mask(crop)

            objects.append(
                {
                    "cls_id": cls_id,
                    "crop": crop,
                    "mask": mask,
                    "source": str(image_path),
                }
            )

            if max_objects is not None and len(objects) >= max_objects:
                return objects

    return objects


def random_background(pairs, out_w, out_h):
    image_path, _ = random.choice(pairs)
    img = cv2.imread(str(image_path))

    if img is None:
        bg = np.ones((out_h, out_w, 3), dtype=np.uint8) * random.randint(180, 240)
        return bg

    bg = cv2.resize(img, (out_w, out_h))
    bg = cv2.GaussianBlur(bg, (31, 31), 0)

    # Làm nền sáng nhẹ để trái nổi rõ hơn
    overlay = np.ones_like(bg) * 220
    bg = cv2.addWeighted(bg, 0.45, overlay, 0.55, 0)

    return bg


def paste_object(canvas, visible_canvas, obj, x, y, scale):
    crop = obj["crop"]
    mask = obj["mask"]

    h, w = crop.shape[:2]

    new_w = max(12, int(w * scale))
    new_h = max(12, int(h * scale))

    crop = cv2.resize(crop, (new_w, new_h))
    mask = cv2.resize(mask, (new_w, new_h))

    canvas_h, canvas_w = canvas.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(canvas_w, x + new_w)
    y2 = min(canvas_h, y + new_h)

    if x2 <= x1 or y2 <= y1:
        return None

    crop_x1 = x1 - x
    crop_y1 = y1 - y
    crop_x2 = crop_x1 + (x2 - x1)
    crop_y2 = crop_y1 + (y2 - y1)

    crop_roi = crop[crop_y1:crop_y2, crop_x1:crop_x2]
    mask_roi = mask[crop_y1:crop_y2, crop_x1:crop_x2]

    alpha = (mask_roi.astype(np.float32) / 255.0)[..., None]

    canvas_roi = canvas[y1:y2, x1:x2]

    blended = crop_roi.astype(np.float32) * alpha + canvas_roi.astype(np.float32) * (1 - alpha)

    canvas[y1:y2, x1:x2] = blended.astype(np.uint8)

    visible_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    visible_mask[y1:y2, x1:x2] = mask_roi

    # Những pixel object mới dán sẽ che object cũ
    visible_canvas[visible_mask > 0] = 0

    # Gán ID tạm cho object mới bằng 255
    visible_canvas[visible_mask > 0] = 255

    return {
        "cls_id": obj["cls_id"],
        "mask": visible_mask,
        "bbox": (x1, y1, x2, y2),
    }


def create_one_synthetic_image(
    objects,
    pairs,
    out_w,
    out_h,
    min_fruits,
    max_fruits,
    overlap_prob,
    min_visible_ratio,
):
    canvas = random_background(pairs, out_w, out_h)

    placed = []
    visible_masks = []

    n = random.randint(min_fruits, max_fruits)

    base_cx = random.randint(int(out_w * 0.25), int(out_w * 0.75))
    base_cy = random.randint(int(out_h * 0.25), int(out_h * 0.75))

    for i in range(n):
        obj = random.choice(objects)

        scale = random.uniform(0.65, 1.25)

        crop_h, crop_w = obj["crop"].shape[:2]
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)

        if i > 0 and random.random() < overlap_prob:
            x = base_cx + random.randint(-int(out_w * 0.18), int(out_w * 0.18)) - new_w // 2
            y = base_cy + random.randint(-int(out_h * 0.18), int(out_h * 0.18)) - new_h // 2
        else:
            x = random.randint(0, max(1, out_w - new_w))
            y = random.randint(0, max(1, out_h - new_h))

        # Trước khi dán object mới, object mới sẽ che các mask cũ.
        crop = obj["crop"]
        mask = obj["mask"]

        new_w = max(12, int(crop.shape[1] * scale))
        new_h = max(12, int(crop.shape[0] * scale))

        crop_resized = cv2.resize(crop, (new_w, new_h))
        mask_resized = cv2.resize(mask, (new_w, new_h))

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(out_w, x + new_w)
        y2 = min(out_h, y + new_h)

        if x2 <= x1 or y2 <= y1:
            continue

        crop_x1 = x1 - x
        crop_y1 = y1 - y
        crop_x2 = crop_x1 + (x2 - x1)
        crop_y2 = crop_y1 + (y2 - y1)

        crop_roi = crop_resized[crop_y1:crop_y2, crop_x1:crop_x2]
        mask_roi = mask_resized[crop_y1:crop_y2, crop_x1:crop_x2]

        new_visible_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        new_visible_mask[y1:y2, x1:x2] = mask_roi

        # Object mới che object cũ
        for old in placed:
            old["visible_mask"][new_visible_mask > 0] = 0

        alpha = (mask_roi.astype(np.float32) / 255.0)[..., None]

        roi = canvas[y1:y2, x1:x2]
        blended = crop_roi.astype(np.float32) * alpha + roi.astype(np.float32) * (1 - alpha)
        canvas[y1:y2, x1:x2] = blended.astype(np.uint8)

        placed.append(
            {
                "cls_id": obj["cls_id"],
                "full_mask": new_visible_mask.copy(),
                "visible_mask": new_visible_mask.copy(),
            }
        )

    labels = []

    for item in placed:
        full_area = (item["full_mask"] > 0).sum()
        visible_area = (item["visible_mask"] > 0).sum()

        if full_area <= 0:
            continue

        visible_ratio = visible_area / full_area

        if visible_ratio < min_visible_ratio:
            continue

        ys, xs = np.where(item["visible_mask"] > 0)

        if len(xs) == 0 or len(ys) == 0:
            continue

        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max())
        y2 = int(ys.max())

        label = xyxy_to_yolo(
            item["cls_id"],
            x1,
            y1,
            x2,
            y2,
            out_w,
            out_h,
        )

        if label is not None:
            labels.append(label)

    return canvas, labels


def copy_original_dataset(src_root, out_root):
    src_root = Path(src_root)
    out_root = Path(out_root)

    for split in ["train", "val", "valid", "test"]:
        src_img_dir = src_root / "images" / split
        src_lbl_dir = src_root / "labels" / split

        if not src_img_dir.exists() or not src_lbl_dir.exists():
            continue

        out_split = "val" if split == "valid" else split

        dst_img_dir = out_root / "images" / out_split
        dst_lbl_dir = out_root / "labels" / out_split

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in src_img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            lbl_path = src_lbl_dir / f"{img_path.stem}.txt"

            if not lbl_path.exists():
                continue

            shutil.copy2(img_path, dst_img_dir / f"orig_{img_path.name}")
            shutil.copy2(lbl_path, dst_lbl_dir / f"orig_{lbl_path.name}")


def infer_class_names(pairs):
    max_cls = 0

    for _, label_path in pairs:
        labels = read_yolo_labels(label_path)

        for label in labels:
            max_cls = max(max_cls, label[0])

    names = [f"class_{i}" for i in range(max_cls + 1)]

    return names


def write_data_yaml(out_root, names):
    out_root = Path(out_root)

    yaml_path = out_root / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {out_root.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")

        for i, name in enumerate(names):
            f.write(f"  {i}: {name}\n")

    return yaml_path


def generate_dataset(args):
    src_root = Path(args.src)
    out_root = Path(args.out)

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)

    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    pairs = find_image_label_pairs(src_root)

    if not pairs:
        raise RuntimeError(
            f"Không tìm thấy cặp images/labels trong {src_root}. "
            f"Hãy kiểm tra dataset YOLO của bạn."
        )

    print(f"Tìm thấy {len(pairs)} ảnh YOLO có label.")

    print("Đang crop object từ dataset gốc...")
    objects = extract_objects(pairs, max_objects=args.max_objects)

    if not objects:
        raise RuntimeError("Không crop được object nào từ YOLO labels.")

    print(f"Đã crop được {len(objects)} object.")

    if args.copy_original:
        print("Đang copy dataset gốc sang dataset mới...")
        copy_original_dataset(src_root, out_root)

    total = args.num_train + args.num_val

    for idx in range(total):
        split = "train" if idx < args.num_train else "val"
        local_idx = idx if split == "train" else idx - args.num_train

        img, labels = create_one_synthetic_image(
            objects=objects,
            pairs=pairs,
            out_w=args.width,
            out_h=args.height,
            min_fruits=args.min_fruits,
            max_fruits=args.max_fruits,
            overlap_prob=args.overlap_prob,
            min_visible_ratio=args.min_visible_ratio,
        )

        img_name = f"overlap_{split}_{local_idx:06d}.jpg"
        lbl_name = f"overlap_{split}_{local_idx:06d}.txt"

        img_path = out_root / "images" / split / img_name
        lbl_path = out_root / "labels" / split / lbl_name

        cv2.imwrite(str(img_path), img)

        with open(lbl_path, "w", encoding="utf-8") as f:
            for line in labels:
                f.write(line + "\n")

        if (idx + 1) % 100 == 0:
            print(f"Đã tạo {idx + 1}/{total} ảnh synthetic overlap...")

    names = infer_class_names(pairs)
    yaml_path = write_data_yaml(out_root, names)

    print("\n===== HOÀN TẤT DATASET CHỒNG CHÉO =====")
    print(f"Dataset mới: {out_root}")
    print(f"File data.yaml: {yaml_path}")
    print("\nBạn có thể train bằng:")
    print(f"yolo detect train model=yolov8n.pt data={yaml_path} imgsz=960 epochs=100 batch=8")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", required=True, help="Dataset YOLO gốc.")
    parser.add_argument("--out", default="yolo_overlap_dataset", help="Dataset YOLO mới.")

    parser.add_argument("--num-train", type=int, default=2000)
    parser.add_argument("--num-val", type=int, default=400)

    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)

    parser.add_argument("--min-fruits", type=int, default=3)
    parser.add_argument("--max-fruits", type=int, default=8)

    parser.add_argument("--overlap-prob", type=float, default=0.85)
    parser.add_argument("--min-visible-ratio", type=float, default=0.18)

    parser.add_argument("--max-objects", type=int, default=None)

    parser.add_argument("--copy-original", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args)