from __future__ import annotations

from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_path: str, tracker_path: str, image_size: int, person_class_id: int):
        self.model = YOLO(model_path, task="detect")
        self.tracker_path = tracker_path
        self.image_size = image_size
        self.person_class_id = person_class_id
        print("YOLO model loaded")

    def detect(self, rgb):
        results = self.model.track(
            source=rgb,
            imgsz=self.image_size,
            classes=[self.person_class_id],
            persist=True,
            tracker=self.tracker_path,
            verbose=False,
        )

        boxes = results[0].boxes
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            track_id = -1 if box.id is None else int(box.id[0])
            detections.append({"track_id": track_id, "conf": conf, "box": (x1, y1, x2, y2)})
        return detections
