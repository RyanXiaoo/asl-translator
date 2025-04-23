import tensorflow as tf
from tensorflow.python.client import device_lib

print("TF built with CUDA?:", tf.test.is_built_with_cuda())
print("All physical devices:", tf.config.list_physical_devices())
print("Detailed device list:") 
for d in device_lib.list_local_devices():
    print(" ", d.name, d.device_type)