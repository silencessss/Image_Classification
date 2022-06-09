import tensorflow as tf
#----------------------------------------------------------#
# GPU Detail
# [TensorFlow 2.0 硬體資源設置](https://hackmd.io/@shaoeChen/ryWIV4vkL)
#----------------------------------------------------------#
def GPU():
    print('#--------Setting GPU-------#')
    print('[INFO - GPU]: ',len(tf.config.list_physical_devices('GPU')))
    #tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print(gpus)
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
    print('[INFO] gpu setting done!')