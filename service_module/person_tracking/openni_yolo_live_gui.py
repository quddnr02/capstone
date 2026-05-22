import time
import numpy as np
import cv2

from primesense import openni2
from ultralytics import YOLO


OPENNI_PATH = "/home/dgdg/astra_sdk/AstraSDK-v2.1.3-94bca0f52e-20210611T023312Z-Linux-aarch64/lib/Plugins/openni2"
MODEL_PATH = "/home/dgdg/person_tracking/yolov8n_ncnn_model"


def main():
    openni2.initialize(OPENNI_PATH)

    dev = openni2.Device.open_any()
    print("Device opened:", dev.get_device_info())

    color_stream = dev.create_color_stream()
    color_stream.start()

    model = YOLO(MODEL_PATH, task="detect")
    print("YOLO model loaded")

    frame_count = 0
    last_time = time.time()
    fps = 0.0

    try:
        while True:
            frame = color_stream.read_frame()

            w = frame.width
            h = frame.height

            data = frame.get_buffer_as_uint8()
            rgb = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)).copy()

            results = model.predict(
                source=rgb,
                imgsz=320,
                classes=[0],
                verbose=False
            )

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            persons = len(results[0].boxes)

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])

                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    bgr,
                    f"person {conf:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

            frame_count += 1
            now = time.time()

            if now - last_time >= 1.0:
                fps = frame_count / (now - last_time)
                print(f"FPS: {fps:.1f} | persons: {persons}")

                frame_count = 0
                last_time = now

            cv2.putText(
                bgr,
                f"FPS: {fps:.1f} | persons: {persons}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow("Astra YOLO Live", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopped")

    finally:
        color_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()