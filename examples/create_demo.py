"""Build a tiny synthetic demo to smoke-test media sampling and pipeline wiring.
This is NOT real-world robot ground-truth training data.
"""
from pathlib import Path
import cv2
import numpy as np
import json


def main():
    root = Path(__file__).resolve().parent
    output = root / "demo.mp4"
    fps, seconds = 25, 12
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    if not writer.isOpened():
        raise RuntimeError("Could not encode demo video")
    try:
        for frame in range(fps * seconds):
            t = frame / fps
            img = np.full((240, 320, 3), 245, dtype=np.uint8)
            cv2.rectangle(img, (210, 145), (295, 215), (165, 190, 220), -1)
            cv2.putText(img, "PALLET", (213, 235), cv2.FONT_HERSHEY_SIMPLEX, .45, (40, 40, 40), 1)
            if t < 4:
                x, y = 45, 150 - int(20 * t / 4)
            elif t < 8:
                x, y = 45 + int(170 * (t - 4) / 4), 130
            else:
                x, y = 215, 130 + int(30 * (t - 8) / 4)
            cv2.rectangle(img, (x, y), (x + 46, y + 35), (60, 120, 175), -1)
            cv2.putText(img, f"T={t:.2f}s", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (20, 20, 20), 1)
            writer.write(img)
    finally:
        writer.release()
    data = {
        "episode_id": "demo_boxes_001",
        "task": "将一个纸箱搬运并放置到托盘上",
        "video_path": "demo.mp4",
        "annotation_fps": 1,
        "subtasks": [
            {"subtask_index": 1, "start_frame_index": 0, "end_frame_index": 4,
             "segment_summary": "机械臂抓取桌面上的纸箱",
             "arm_1": {"description": "移动到纸箱处并抓取", "outcome": "抓取完成"},
             "arm_2": {"description": "无动作", "outcome": "无"}},
            {"subtask_index": 2, "start_frame_index": 4, "end_frame_index": 8,
             "segment_summary": "机械臂将纸箱搬运至托盘上方",
             "arm_1": {"description": "持箱移动到托盘区域", "outcome": "纸箱位于托盘上方"},
             "arm_2": {"description": "无动作", "outcome": "无"}},
            {"subtask_index": 3, "start_frame_index": 8, "end_frame_index": 12,
             "segment_summary": "机械臂将纸箱放在托盘上",
             "arm_1": {"description": "下降并放置纸箱", "outcome": "完成放置"},
             "arm_2": {"description": "无动作", "outcome": "无"}},
        ],
    }
    (root / "episode.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Generated", output, "and", root / "episode.json")


if __name__ == "__main__":
    main()
