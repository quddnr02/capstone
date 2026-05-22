from primesense import openni2

OPENNI_PATH = "/home/dgdg/astra_sdk/AstraSDK-v2.1.3-94bca0f52e-20210611T023312Z-Linux-aarch64/lib/Plugins/openni2"

openni2.initialize(OPENNI_PATH)

dev = openni2.Device.open_any()
info = dev.get_device_info()

print("Device opened")
print("name:", info.name)
print("uri:", info.uri)
print("vendor:", info.vendor)

openni2.unload()