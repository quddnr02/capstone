import time
import math
import numpy as np
import cv2

from primesense import openni2
from ultralytics import YOLO


OPENNI_PATH = "/home/dgdg/astra_sdk/AstraSDK-v2.1.3-94bca0f52e-20210611T023312Z-Linux-aarch64/lib/Plugins/openni2"
MODEL_PATH = "/home/dgdg/person_tracking/yolov8n_ncnn_model"
TRACKER_PATH = "/home/dgdg/person_tracking/bytetrack_custom.yaml"

ROI_HALF = 5
MIN_DEPTH_MM = 300
MAX_DEPTH_MM = 6000
COLOR_HFOV_DEG = 60.0

MAX_LOST_FRAMES = 60
RECONNECT_MAX_CENTER_DIST = 120
RECONNECT_MAX_DEPTH_DIFF_M = 0.8
RECONNECT_MAX_ANGLE_DIFF_DEG = 15.0

COLOR_BOX = (0, 180, 0)
COLOR_TARGET_BOX = (255, 0, 0)
COLOR_CENTER = (0, 0, 255)
COLOR_TEXT = (0, 0, 0)
COLOR_TRACK_TEXT = (255, 0, 0)
COLOR_TARGET_TEXT = (0, 0, 255)
COLOR_BG = (220, 220, 220)


target_track_id = None
target_lost_count = 0
last_target_center = None
last_target_depth_m = None
last_target_angle_x = None


def median_depth_mm(depth_img, cx, cy):
    h, w = depth_img.shape

    x1 = max(0, cx - ROI_HALF)
    x2 = min(w, cx + ROI_HALF + 1)
    y1 = max(0, cy - ROI_HALF)
    y2 = min(h, cy + ROI_HALF + 1)

    roi = depth_img[y1:y2, x1:x2]
    valid = roi[(roi >= MIN_DEPTH_MM) & (roi <= MAX_DEPTH_MM)]

    if valid.size == 0:
        return None

    return float(np.median(valid))


def calc_angle_x_deg(cx, image_width):
    center_x = image_width / 2.0
    norm_x = (cx - center_x) / center_x
    return float(norm_x * (COLOR_HFOV_DEG / 2.0))


def draw_text_with_bg(img, text, pos, font_scale=0.5, text_color=COLOR_TEXT):
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    cv2.rectangle(
        img,
        (x - 2, y - th - baseline - 2),
        (x + tw + 2, y + baseline + 2),
        COLOR_BG,
        -1
    )

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness
    )


def center_distance(c1, c2):
    if c1 is None or c2 is None:
        return 999999.0

    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return math.sqrt(dx * dx + dy * dy)


def make_detection(box, cw, ch, dw, dh, depth):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    conf = float(box.conf[0])

    if box.id is None:
        track_id = -1
    else:
        track_id = int(box.id[0])

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    dx = int(cx * dw / cw)
    dy = int(cy * dh / ch)

    depth_mm = median_depth_mm(depth, dx, dy)
    depth_m = None if depth_mm is None else depth_mm / 1000.0
    angle_x = calc_angle_x_deg(cx, cw)

    return {
        "track_id": track_id,
        "conf": conf,
        "box": (x1, y1, x2, y2),
        "center": (cx, cy),
        "depth_m": depth_m,
        "angle_x": angle_x,
    }


def select_initial_target(detections, image_width):
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


def reconnect_target(detections):
    global last_target_center, last_target_depth_m, last_target_angle_x

    if not detections or last_target_center is None:
        return None

    best = None
    best_score = 999999.0

    for d in detections:
        dist = center_distance(d["center"], last_target_center)
        angle_diff = abs(d["angle_x"] - last_target_angle_x) if last_target_angle_x is not None else 0.0

        if d["depth_m"] is not None and last_target_depth_m is not None:
            depth_diff = abs(d["depth_m"] - last_target_depth_m)
        else:
            depth_diff = 0.0

        if dist > RECONNECT_MAX_CENTER_DIST:
            continue

        if angle_diff > RECONNECT_MAX_ANGLE_DIFF_DEG:
            continue

        if depth_diff > RECONNECT_MAX_DEPTH_DIFF_M:
            continue

        score = dist + angle_diff * 4.0 + depth_diff * 80.0

        if score < best_score:
            best_score = score
            best = d

    return best


def update_target_lock(detections, image_width):
    global target_track_id
    global target_lost_count
    global last_target_center
    global last_target_depth_m
    global last_target_angle_x

    target = None

    if target_track_id is None:
        target = select_initial_target(detections, image_width)
        if target is not None:
            target_track_id = target["track_id"]
            target_lost_count = 0

    else:
        for d in detections:
            if d["track_id"] == target_track_id and d["track_id"] != -1:
                target = d
                target_lost_count = 0
                break

        if target is None:
            reconnected = reconnect_target(detections)

            if reconnected is not None:
                target = reconnected
                target_track_id = target["track_id"]
                target_lost_count = 0
                print(f"TARGET RECONNECTED -> new track_id={target_track_id}")
            else:
                target_lost_count += 1

                if target_lost_count > MAX_LOST_FRAMES:
                    print("TARGET LOST -> reset target")
                    target_track_id = None
                    target_lost_count = 0
                    last_target_center = None
                    last_target_depth_m = None
                    last_target_angle_x = None

    if target is not None:
        last_target_center = target["center"]
        if target["depth_m"] is not None:
            last_target_depth_m = target["depth_m"]
        last_target_angle_x = target["angle_x"]

    return target


def main():
    global target_track_id
    global target_lost_count

    openni2.initialize(OPENNI_PATH)

    dev = openni2.Device.open_any()
    print("Device opened:", dev.get_device_info())

    color_stream = dev.create_color_stream()
    depth_stream = dev.create_depth_stream()

    color_stream.start()
    depth_stream.start()

    model = YOLO(MODEL_PATH, task="detect")
    print("YOLO model loaded")

    frame_count = 0
    last_time = time.time()
    fps = 0.0

    try:
        while True:
            color_frame = color_stream.read_frame()
            depth_frame = depth_stream.read_frame()

            cw, ch = color_frame.width, color_frame.height
            dw, dh = depth_frame.width, depth_frame.height

            color_data = color_frame.get_buffer_as_uint8()
            depth_data = depth_frame.get_buffer_as_uint16()

            rgb = np.frombuffer(color_data, dtype=np.uint8).reshape((ch, cw, 3)).copy()
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape((dh, dw)).copy()

            results = model.track(
                source=rgb,
                imgsz=320,
                classes=[0],
                persist=True,
                tracker=TRACKER_PATH,
                verbose=False
            )

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            boxes = results[0].boxes
            detections = [
                make_detection(box, cw, ch, dw, dh, depth)
                for box in boxes
            ]

            target = update_target_lock(detections, cw)

            for d in detections:
                x1, y1, x2, y2 = d["box"]
                cx, cy = d["center"]
                track_id = d["track_id"]
                conf = d["conf"]
                depth_m = d["depth_m"]
                angle_x = d["angle_x"]

                is_target = target is not None and d is target

                box_color = COLOR_TARGET_BOX if is_target else COLOR_BOX
                text_color = COLOR_TARGET_TEXT if is_target else COLOR_TRACK_TEXT

                cv2.rectangle(bgr, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(bgr, (cx, cy), 5, COLOR_CENTER, -1)

                if is_target:
                    label1 = f"TARGET | ID {track_id} | conf {conf:.2f}"
                else:
                    label1 = f"ID {track_id} | conf {conf:.2f}"

                if depth_m is None:
                    label2 = f"depth invalid | angle {angle_x:.1f} deg"
                else:
                    label2 = f"depth {depth_m:.2f} m | angle {angle_x:.1f} deg"

                draw_text_with_bg(
                    bgr,
                    label1,
                    (x1, max(20, y1 - 28)),
                    font_scale=0.5,
                    text_color=text_color
                )

                draw_text_with_bg(
                    bgr,
                    label2,
                    (x1, max(20, y1 - 8)),
                    font_scale=0.5,
                    text_color=COLOR_TEXT
                )

            if target is not None:
                if target["depth_m"] is None:
                    print(
                        f"TARGET id={target['track_id']} "
                        f"angle_x={target['angle_x']:.1f} deg "
                        f"depth=invalid"
                    )
                else:
                    print(
                        f"TARGET id={target['track_id']} "
                        f"angle_x={target['angle_x']:.1f} deg "
                        f"depth={target['depth_m']:.2f} m"
                    )
            else:
                print(
                    f"TARGET none "
                    f"last_id={target_track_id} "
                    f"lost={target_lost_count}"
                )

            frame_count += 1
            now = time.time()

            if now - last_time >= 1.0:
                fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now

            cv2.rectangle(bgr, (5, 5), (410, 38), COLOR_BG, -1)

            status = "LOCKED" if target is not None else "SEARCHING"
            cv2.putText(
                bgr,
                f"FPS {fps:.1f} | persons {len(detections)} | target {status} | id {target_track_id}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_TEXT,
                2
            )

            cv2.imshow("Astra YOLO Target Lock", bgr)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                print("MANUAL RESET TARGET")
                target_track_id = None
                target_lost_count = 0

    except KeyboardInterrupt:
        print("\nStopped")

    finally:
        color_stream.stop()
        depth_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()