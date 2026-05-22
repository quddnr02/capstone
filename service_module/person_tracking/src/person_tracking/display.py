from __future__ import annotations

import cv2


class Display:
    COLOR_BOX = (0, 180, 0)
    COLOR_TARGET_BOX = (255, 0, 0)
    COLOR_CENTER = (0, 0, 255)
    COLOR_TEXT = (0, 0, 0)
    COLOR_TRACK_TEXT = (255, 0, 0)
    COLOR_TARGET_TEXT = (0, 0, 255)
    COLOR_BG = (220, 220, 220)

    def draw_text_with_bg(self, img, text, pos, font_scale=0.5, text_color=(0, 0, 0)):
        x, y = pos
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, (x - 2, y - th - baseline - 2), (x + tw + 2, y + baseline + 2), self.COLOR_BG, -1)
        cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

    def render(self, bgr, detections, target, fps, target_track_id):
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            cx, cy = d["center"]
            is_target = target is not None and d is target
            box_color = self.COLOR_TARGET_BOX if is_target else self.COLOR_BOX
            text_color = self.COLOR_TARGET_TEXT if is_target else self.COLOR_TRACK_TEXT
            cv2.rectangle(bgr, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(bgr, (cx, cy), 5, self.COLOR_CENTER, -1)
            label1 = f"TARGET | ID {d['track_id']} | conf {d['conf']:.2f}" if is_target else f"ID {d['track_id']} | conf {d['conf']:.2f}"
            label2 = f"depth invalid | angle {d['angle_x']:.1f} deg" if d["depth_m"] is None else f"depth {d['depth_m']:.2f} m | angle {d['angle_x']:.1f} deg"
            self.draw_text_with_bg(bgr, label1, (x1, max(20, y1 - 28)), 0.5, text_color)
            self.draw_text_with_bg(bgr, label2, (x1, max(20, y1 - 8)), 0.5, self.COLOR_TEXT)

        cv2.rectangle(bgr, (5, 5), (410, 38), self.COLOR_BG, -1)
        status = "LOCKED" if target is not None else "SEARCHING"
        cv2.putText(
            bgr,
            f"FPS {fps:.1f} | persons {len(detections)} | target {status} | id {target_track_id}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.COLOR_TEXT,
            2,
        )
        cv2.imshow("Astra YOLO Target Lock", bgr)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        cv2.destroyAllWindows()
