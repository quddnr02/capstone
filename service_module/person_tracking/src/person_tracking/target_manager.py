from __future__ import annotations

import math


class TargetManager:
    def __init__(self, config: dict):
        self.max_lost_frames = config["max_lost_frames"]
        self.reconnect_max_center_dist = config["reconnect_max_center_dist"]
        self.reconnect_max_depth_diff_m = config["reconnect_max_depth_diff_m"]
        self.reconnect_max_angle_diff_deg = config["reconnect_max_angle_diff_deg"]

        self._target_track_id = None
        self.target_lost_count = 0
        self.last_target_center = None
        self.last_target_depth_m = None
        self.last_target_angle_x = None
        self.reconnected_this_frame = False

    @property
    def target_track_id(self):
        return self._target_track_id

    def reset(self):
        self._target_track_id = None
        self.target_lost_count = 0
        self.last_target_center = None
        self.last_target_depth_m = None
        self.last_target_angle_x = None
        self.reconnected_this_frame = False

    def center_distance(self, c1, c2):
        if c1 is None or c2 is None:
            return 999999.0
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return math.sqrt(dx * dx + dy * dy)

    def select_initial_target(self, detections, image_width):
        if not detections:
            return None
        image_center_x = image_width / 2.0
        valid = [d for d in detections if d["depth_m"] is not None]
        candidates = valid if valid else detections

        def score(d):
            cx, _ = d["center"]
            center_score = abs(cx - image_center_x)
            depth_score = d["depth_m"] if d["depth_m"] is not None else 10.0
            return center_score + depth_score * 20.0

        return min(candidates, key=score)

    def reconnect_target(self, detections):
        if not detections or self.last_target_center is None:
            return None
        best = None
        best_score = 999999.0
        for d in detections:
            dist = self.center_distance(d["center"], self.last_target_center)
            angle_diff = abs(d["angle_x"] - self.last_target_angle_x) if self.last_target_angle_x is not None else 0.0
            if d["depth_m"] is not None and self.last_target_depth_m is not None:
                depth_diff = abs(d["depth_m"] - self.last_target_depth_m)
            else:
                depth_diff = 0.0
            if dist > self.reconnect_max_center_dist or angle_diff > self.reconnect_max_angle_diff_deg or depth_diff > self.reconnect_max_depth_diff_m:
                continue
            score = dist + angle_diff * 4.0 + depth_diff * 80.0
            if score < best_score:
                best_score = score
                best = d
        return best

    def update(self, detections: list[dict], image_width: int):
        self.reconnected_this_frame = False
        target = None
        if self._target_track_id is None:
            target = self.select_initial_target(detections, image_width)
            if target is not None:
                self._target_track_id = target["track_id"]
                self.target_lost_count = 0
        else:
            for d in detections:
                if d["track_id"] == self._target_track_id and d["track_id"] != -1:
                    target = d
                    self.target_lost_count = 0
                    break
            if target is None:
                reconnected = self.reconnect_target(detections)
                if reconnected is not None:
                    target = reconnected
                    self._target_track_id = target["track_id"]
                    self.target_lost_count = 0
                    self.reconnected_this_frame = True
                    print(f"TARGET RECONNECTED -> new track_id={self._target_track_id}")
                else:
                    self.target_lost_count += 1
                    if self.target_lost_count > self.max_lost_frames:
                        print("TARGET LOST -> reset target")
                        self.reset()

        if target is not None:
            self.last_target_center = target["center"]
            if target["depth_m"] is not None:
                self.last_target_depth_m = target["depth_m"]
            self.last_target_angle_x = target["angle_x"]
        return target
