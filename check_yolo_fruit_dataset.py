from pathlib import Path

ROOT = Path("/mnt/d/merged_fruit_36_clean_v2")
DATASET = ROOT / "yolo_fruit_dataset"

for split in ["train", "val"]:
    img_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    images = list(img_dir.glob("*.jpg"))

    missing = []
    empty = []
    invalid = []
    total_boxes = 0

    for img in images:
        label = label_dir / f"{img.stem}.txt"

        if not label.exists():
            missing.append(img.name)
            continue

        text = label.read_text(encoding="utf-8").strip()

        if text == "":
            empty.append(label.name)
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            parts = line.strip().split()

            if len(parts) != 5:
                invalid.append((label.name, line_no, line))
                continue

            try:
                cls_id = int(parts[0])
                vals = [float(x) for x in parts[1:]]
            except Exception:
                invalid.append((label.name, line_no, line))
                continue

            if cls_id != 0:
                invalid.append((label.name, line_no, line))
                continue

            if not all(0 <= v <= 1 for v in vals):
                invalid.append((label.name, line_no, line))
                continue

            total_boxes += 1

    print(f"\n===== {split.upper()} =====")
    print("Images:", len(images))
    print("Missing labels:", len(missing))
    print("Empty labels:", len(empty))
    print("Invalid labels:", len(invalid))
    print("Total boxes:", total_boxes)

    if empty:
        print("Ví dụ label rỗng:", empty[:10])

    if invalid:
        print("Ví dụ label lỗi:", invalid[:10])

print("\nNếu Empty labels còn nhiều, bạn chưa vẽ bbox xong.")
