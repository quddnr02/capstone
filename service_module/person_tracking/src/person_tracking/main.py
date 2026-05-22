from __future__ import annotations

import time
from pathlib import Path

import cv2
import yaml

from person_tracking.camera import AstraCamera
from person_tracking.detector import PersonDetector
from person_tracking.display import Display
from person_tracking.measure import Measure
from person_tracking.target_manager import TargetManager


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    root = config_path.parent.parent
    config["paths"]["model"] = str((root / config["paths"]["model"]).resolve())
    config["paths"]["tracker"] = str((root / config["paths"]["tracker"]).resolve())
    return config


def main():
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "config" / "config.yaml")

    camera = AstraCamera(config["paths"]["openni"])
    detector = PersonDetector(config["paths"]["model"], config["paths"]["tracker"], config["detection"]["image_size"], config["detection"]["person_class_id"])
    measure = Measure(config["measure"])
    target_manager = TargetManager(config["target"])
    display = Display()

    frame_count = 0
    last_time = time.time()
    fps = 0.0

    try:
        camera.start()
        while True:
            rgb, depth = camera.read()
            detections = detector.detect(rgb)
            detections = measure.add_measurements(detections, rgb.shape, depth.shape, depth)
            target = target_manager.update(detections, rgb.shape[1])

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            key = display.render(bgr, detections, target, fps, target_manager.target_track_id)

            if target is not None:
                if target["depth_m"] is None:
                    print(f"TARGET id={target['track_id']} angle_x={target['angle_x']:.1f} deg depth=invalid")
                else:
                    print(f"TARGET id={target['track_id']} angle_x={target['angle_x']:.1f} deg depth={target['depth_m']:.2f} m")
            else:
                print(f"TARGET none last_id={target_manager.target_track_id} lost={target_manager.target_lost_count}")

            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now

            if key == ord("q"):
                break
            if key == ord("r"):
                print("MANUAL RESET TARGET")
                target_manager.reset()

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        camera.close()
        display.close()


if __name__ == "__main__":
    main()
