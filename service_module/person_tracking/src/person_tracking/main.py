from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import yaml

from person_tracking.camera import AstraCamera
from person_tracking.can_sender import TargetCanSender
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Person tracking with optional CAN output")
    parser.add_argument("--no-can", action="store_true", help="Disable CAN transmission")
    parser.add_argument("--can-channel", default="can0", help="SocketCAN channel name")
    parser.add_argument("--can-bitrate", type=int, default=500000, help="CAN bitrate")
    parser.add_argument("--can-send-hz", type=float, default=10.0, help="CAN max send rate")
    parser.add_argument("--can-debug", action="store_true", help="Print CAN payload debug logs")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "config" / "config.yaml")

    camera = AstraCamera(config["paths"]["openni"])
    detector = PersonDetector(config["paths"]["model"], config["paths"]["tracker"], config["detection"]["image_size"], config["detection"]["person_class_id"])
    measure = Measure(config["measure"])
    target_manager = TargetManager(config["target"])
    display = Display()
    can_sender = TargetCanSender(
        channel=args.can_channel,
        interface="socketcan",
        bitrate=args.can_bitrate,
        arbitration_id=0x101,
        send_hz=args.can_send_hz,
        enabled=not args.no_can,
        debug=args.can_debug,
    )

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
                can_sender.send_target_state(
                    track_id=target.get("track_id"),
                    depth_m=target.get("depth_m"),
                    angle_x_deg=target.get("angle_x"),
                    confidence=target.get("conf"),
                    reconnected=target_manager.reconnected_this_frame,
                )
            else:
                print(f"TARGET none last_id={target_manager.target_track_id} lost={target_manager.target_lost_count}")
                can_sender.send_target_state(
                    track_id=None,
                    depth_m=None,
                    angle_x_deg=None,
                    confidence=None,
                    reconnected=False,
                )

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
        can_sender.close()
        camera.close()
        display.close()


if __name__ == "__main__":
    main()
