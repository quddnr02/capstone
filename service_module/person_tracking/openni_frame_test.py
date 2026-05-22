from primesense import openni2
import numpy as np
import cv2
import time

OPENNI_PATH = "/home/dgdg/astra_sdk/AstraSDK-v2.1.3-94bca0f52e-20210611T023312Z-Linux-aarch64/lib/Plugins/openni2"

openni2.initialize(OPENNI_PATH)

dev = openni2.Device.open_any()
print("Device:", dev.get_device_info())

depth_stream = dev.create_depth_stream()
color_stream = dev.create_color_stream()

depth_stream.start()
color_stream.start()

time.sleep(1.0)

for i in range(30):
    depth_frame = depth_stream.read_frame()
    color_frame = color_stream.read_frame()

    dw, dh = depth_frame.width, depth_frame.height
    cw, ch = color_frame.width, color_frame.height

    depth_data = depth_frame.get_buffer_as_uint16()
    color_data = color_frame.get_buffer_as_uint8()

    depth = np.frombuffer(depth_data, dtype=np.uint16).reshape((dh, dw))
    color = np.frombuffer(color_data, dtype=np.uint8).reshape((ch, cw, 3))

    print(f"[{i}] depth={depth.shape} min={depth.min()} max={depth.max()} | color={color.shape}")

    if i == 0:
        cv2.imwrite("openni_color.png", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        depth_vis = depth.copy()
        depth_vis[depth_vis > 5000] = 5000
        depth_vis = (depth_vis / 5000 * 255).astype(np.uint8)
        cv2.imwrite("openni_depth.png", depth_vis)

depth_stream.stop()
color_stream.stop()
openni2.unload()

print("saved openni_color.png and openni_depth.png")