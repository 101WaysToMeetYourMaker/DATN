from ultralytics import YOLO
from collections import defaultdict, Counter
from pathlib import Path
import argparse
import cv2
import json
import math


class TrackMerger:
    """
    Gộp các track bị đổi ID sau khi trái cây bị che khuất.
    Ví dụ: cùng 1 trái ban đầu ID=3, sau khi bị che xuất hiện lại thành ID=8.
    Nếu cùng class, gần vị trí cũ, và mất không quá max_gap frame thì gộp lại.
    """

    def __init__(self, max_gap=30, max_center_dist=90):
        self.max_gap = max_gap
        self.max_center_dist = max_center_dist
        self.last_info = {}
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, old_id, new_id):
        root_old = self.find(old_id)
        root_new = self.find(new_id)

        if root_old != root_new:
            self.parent[root_new] = root_old

    def update(self, track_id, class_name, box_xyxy, frame_idx):
        x1, y1, x2, y2 = box_xyxy

        cx = float((x1 + x2) / 2)
        cy = float((y1 + y2) / 2)

        for old_id, info in list(self.last_info.items()):
            if old_id == track_id:
                continue

            if info["class_name"] != class_name:
                continue

            gap = frame_idx - info["frame_idx"]

            if gap <= 0 or gap > self.max_gap:
                continue

            dist = math.hypot(cx - info["cx"], cy - info["cy"])

            if dist <= self.max_center_dist:
                self.union(old_id, track_id)
                break

        self.last_info[track_id] = {
            "class_name": class_name,
            "cx": cx,
            "cy": cy,
            "frame_idx": frame_idx,
        }

        return self.find(track_id)


class StableFruitCounter:
    """
    Chỉ đếm khi một track xuất hiện đủ số frame ổn định.
    Mỗi global_id chỉ được đếm 1 lần.
    """

    def __init__(self, min_stable_frames=5):
        self.min_stable_frames = min_stable_frames

        self.track_seen_frames = defaultdict(int)
        self.track_class_votes = defaultdict(list)
        self.counted_ids = set()
        self.final_counts = Counter()

    def update(self, global_id, class_name):
        self.track_seen_frames[global_id] += 1
        self.track_class_votes[global_id].append(class_name)

        just_counted = False
        counted_class = None

        if (
            self.track_seen_frames[global_id] >= self.min_stable_frames
            and global_id not in self.counted_ids
        ):
            stable_class = Counter(self.track_class_votes[global_id]).most_common(1)[0][0]

            self.final_counts[stable_class] += 1
            self.counted_ids.add(global_id)

            just_counted = True
            counted_class = stable_class

        return just_counted, counted_class


def draw_counts(frame, counts):
    y = 30

    cv2.rectangle(frame, (10, 10), (390, 45 + 28 * max(1, len(counts))), (0, 0, 0), -1)

    cv2.putText(
        frame,
        "FRUIT COUNTS",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    y += 30

    if not counts:
        cv2.putText(
            frame,
            "No stable fruit counted yet",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return frame

    for class_name, count in counts.most_common():
        text = f"{class_name}: {count}"

        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        y += 28

    return frame


def count_fruits_in_video(
    model_path,
    video_path,
    output_path,
    tracker_path,
    conf=0.15,
    iou=0.80,
    imgsz=960,
    min_stable_frames=5,
    max_gap=30,
    max_center_dist=90,
    show=False,
):
    model = YOLO(model_path)

    video_path = str(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    merger = TrackMerger(
        max_gap=max_gap,
        max_center_dist=max_center_dist,
    )

    counter = StableFruitCounter(
        min_stable_frames=min_stable_frames,
    )

    results = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker=tracker_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )

    for frame_idx, result in enumerate(results):
        frame = result.orig_img.copy()

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes

            track_ids = boxes.id.cpu().numpy().astype(int)
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            xyxy_list = boxes.xyxy.cpu().numpy()

            for track_id, cls_id, det_conf, xyxy in zip(
                track_ids,
                class_ids,
                confs,
                xyxy_list,
            ):
                if det_conf < conf:
                    continue

                class_name = result.names[int(cls_id)]

                global_id = merger.update(
                    track_id=int(track_id),
                    class_name=class_name,
                    box_xyxy=xyxy,
                    frame_idx=frame_idx,
                )

                just_counted, counted_class = counter.update(
                    global_id=global_id,
                    class_name=class_name,
                )

                x1, y1, x2, y2 = map(int, xyxy)

                label = f"ID:{global_id} {class_name} {det_conf:.2f}"

                if just_counted:
                    label += " COUNTED"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

        frame = draw_counts(frame, counter.final_counts)

        writer.write(frame)

        if show:
            cv2.imshow("Fruit Counting", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    writer.release()
    cv2.destroyAllWindows()

    result_json_path = output_path.with_suffix(".json")

    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video": video_path,
                "model": str(model_path),
                "output_video": str(output_path),
                "counts": dict(counter.final_counts),
                "total": int(sum(counter.final_counts.values())),
                "settings": {
                    "conf": conf,
                    "iou": iou,
                    "imgsz": imgsz,
                    "min_stable_frames": min_stable_frames,
                    "max_gap": max_gap,
                    "max_center_dist": max_center_dist,
                    "tracker": tracker_path,
                },
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    print("\n===== KẾT QUẢ ĐẾM CUỐI CÙNG =====")

    for class_name, count in counter.final_counts.most_common():
        print(f"{class_name}: {count}")

    print(f"\nTổng số trái: {sum(counter.final_counts.values())}")
    print(f"Video output: {output_path}")
    print(f"JSON output: {result_json_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Đường dẫn model .pt. Ví dụ: runs/detect/train/weights/best.pt",
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Đường dẫn video input. Ví dụ: videos/test.mp4",
    )

    parser.add_argument(
        "--output",
        default="outputs/fruit_counted.mp4",
        help="Đường dẫn video output.",
    )

    parser.add_argument(
        "--tracker",
        default="configs/fruit_botsort_occlusion.yaml",
        help="Đường dẫn file tracker yaml.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Ngưỡng confidence. Video bị che khuất nên để thấp khoảng 0.10 - 0.25.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.80,
        help="Ngưỡng IoU NMS. Trái chồng chéo nên để cao hơn mặc định, khoảng 0.75 - 0.85.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Kích thước ảnh inference. Trái nhỏ/chồng chéo nên thử 960 hoặc 1280.",
    )

    parser.add_argument(
        "--min-stable-frames",
        type=int,
        default=5,
        help="Track phải xuất hiện ít nhất bao nhiêu frame mới được đếm.",
    )

    parser.add_argument(
        "--max-gap",
        type=int,
        default=30,
        help="Số frame tối đa cho phép mất dấu để gộp lại cùng một trái.",
    )

    parser.add_argument(
        "--max-center-dist",
        type=int,
        default=90,
        help="Khoảng cách tâm tối đa để gộp track bị đổi ID.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Hiển thị video khi chạy.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    count_fruits_in_video(
        model_path=args.model,
        video_path=args.video,
        output_path=args.output,
        tracker_path=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        min_stable_frames=args.min_stable_frames,
        max_gap=args.max_gap,
        max_center_dist=args.max_center_dist,
        show=args.show,
    )