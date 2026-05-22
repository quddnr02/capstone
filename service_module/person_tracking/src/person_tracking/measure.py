from __future__ import annotations

import numpy as np


class Measure:
    def __init__(self, config: dict):
        self.roi_half = config["roi_half"]
        self.min_depth_mm = config["min_depth_mm"]
        self.max_depth_mm = config["max_depth_mm"]
        self.horizontal_fov_deg = config["horizontal_fov_deg"]

    def median_depth_mm(self, depth_img, cx, cy):
        h, w = depth_img.shape
        x1 = max(0, cx - self.roi_half)
        x2 = min(w, cx + self.roi_half + 1)
        y1 = max(0, cy - self.roi_half)
        y2 = min(h, cy + self.roi_half + 1)

        roi = depth_img[y1:y2, x1:x2]
        valid = roi[(roi >= self.min_depth_mm) & (roi <= self.max_depth_mm)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def calc_angle_x_deg(self, cx, image_width):
        center_x = image_width / 2.0
        norm_x = (cx - center_x) / center_x
        return float(norm_x * (self.horizontal_fov_deg / 2.0))

    def add_measurements(self, detections, color_shape, depth_shape, depth):
        ch, cw = color_shape[:2]
        dh, dw = depth_shape[:2]
        measured = []
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            dx = int(cx * dw / cw)
            dy = int(cy * dh / ch)
            depth_mm = self.median_depth_mm(depth, dx, dy)
            depth_m = None if depth_mm is None else depth_mm / 1000.0
            angle_x = self.calc_angle_x_deg(cx, cw)
            nd = dict(d)
            nd.update({"center": (cx, cy), "depth_m": depth_m, "angle_x": angle_x})
            measured.append(nd)
        return measured
