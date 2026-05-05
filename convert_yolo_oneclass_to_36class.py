from pathlib import Path
import shutil
import re
import yaml

SRC = Path("yolo_fruit_dataset_semiauto")
OUT = Path("yolo_fruit_dataset_36class")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Bắt class theo mẫu: tenfruit_fresh_... hoặc tenfruit_rotten_...
# Ví dụ: apple_fresh_01_xxx.jpg -> apple_fresh
# Ví dụ: bellpepper_rotten_03_xxx.jpg -> bellpepper_rotten
pattern = re.compile(r"^(.+?_(?:fresh|rotten))[_\-].*", re.IGNORECASE)

if OUT.exists():
    shutil.rmtree(OUT)

classes = set()
image_items = []

for img in SRC.rglob("*"):
    if img.suffix.lower() not in IMG_EXTS:
        continue

    m = pattern.match(img.stem)
    if not m:
        continue

    cls_name = m.group(1).lower()
    classes.add(cls_name)
    image_items.append((img, cls_name))

classes = sorted(classes)
class_to_id = {name: i for i, name in enumerate(classes)}

print("===== DANH SÁCH CLASS TÌM ĐƯỢC =====")
for i, name in enumerate(classes):
    print(i, name)

print("\nTổng class:", len(classes))
print("Tổng ảnh nhận diện được class từ tên file:", len(image_items))

if len(classes) == 0:
    raise SystemExit("Không tìm thấy class nào từ tên file. Kiểm tra lại tên ảnh.")

converted = 0
missing_label = 0
bad_label = 0

for img, cls_name in image_items:
    rel = img.relative_to(SRC)

    # Tìm label YOLO tương ứng
    # Trường hợp chuẩn: images/train/a.jpg -> labels/train/a.txt
    parts = list(rel.parts)
    if "images" in parts:
        idx = parts.index("images")
        label_parts = parts.copy()
        label_parts[idx] = "labels"
        src_label = SRC.joinpath(*label_parts).with_suffix(".txt")
    else:
        src_label = img.with_suffix(".txt")

    if not src_label.exists():
        missing_label += 1
        continue

    out_img = OUT / rel
    out_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, out_img)

    out_label_rel = list(rel.parts)
    if "images" in out_label_rel:
        idx = out_label_rel.index("images")
        out_label_rel[idx] = "labels"
        out_label = OUT.joinpath(*out_label_rel).with_suffix(".txt")
    else:
        out_label = OUT / rel.with_suffix(".txt")

    out_label.parent.mkdir(parents=True, exist_ok=True)

    cls_id = class_to_id[cls_name]
    new_lines = []

    for line in src_label.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue

        parts_line = line.split()

        # YOLO detect phải có: class x_center y_center width height
        if len(parts_line) < 5:
            bad_label += 1
            continue

        # Giữ nguyên bbox, chỉ thay class id
        new_line = " ".join([str(cls_id)] + parts_line[1:])
        new_lines.append(new_line)

    out_label.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
    converted += 1

data = {
    "path": str(OUT.resolve()),
    "train": "images/train",
    "val": "images/val",
    "nc": len(classes),
    "names": {i: name for i, name in enumerate(classes)}
}

with open(OUT / "data.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

print("\n===== KẾT QUẢ =====")
print("Dataset mới:", OUT)
print("Số ảnh đã convert:", converted)
print("Số ảnh thiếu label:", missing_label)
print("Số dòng label lỗi:", bad_label)
print("File YAML:", OUT / "data.yaml")
