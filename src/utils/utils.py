
import os
import platform
import tensorflow as tf

from tensorflow.keras.models import Model
from sdks.novavision.src.base.download import Download
from tensorflow.keras.layers import GlobalAveragePooling2D
from sdks.novavision.src.base.application import Application

weight_caption_path = '/storage/model.h5'
weight_url = 'https://drive.google.com/file/d/1omwaXt8-Hnq4YLNizl3PFy0OE7EyzoOh/view?usp=sharing'
feature_weight_path = '/storage/efficientnetb0_notop.h5'
feature_weight_url = 'https://drive.google.com/file/d/1EeKtWayy7NHqozqeb4-dARX52o-jxq-6/view?usp=sharing'
tokenizer_path = '/storage/tokenizer.json'
tokenizer_url = 'https://drive.google.com/file/d/1rym52IjLNjwE--V1nHH65YuqDf6eP4Cl/view?usp=sharing'
output_directory = '/storage/'

def select_device(device='', batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f'TensorFlow Python-{platform.python_version()} tensorflow-{tf.__version__} '
    device = str(device).strip().lower().replace('gpu:', '').replace('none', '')
    cpu = device == 'cpu'

    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device

    physical_devices = tf.config.list_physical_devices('GPU')

    if not cpu and physical_devices:
        devices = device.split(',') if device else [str(i) for i in range(len(physical_devices))]
        n = len(devices)
        if n > 1 and batch_size > 0:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'

        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            with tf.device(f'/device:GPU:{d}'):
                device_name = tf.test.gpu_device_name()
                mem = tf.config.experimental.get_memory_info(f'GPU:{d}')
                s += f"{'' if i == 0 else space}GPU:{d} ({device_name}, {mem['total'] / (1 << 20):.0f}MiB)\n"

        tf.config.set_visible_devices([physical_devices[int(d)] for d in devices], 'GPU')
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)
        arg = f'/device:GPU:{devices[0]}'
    else:
        s += 'CPU\n'
        arg = '/device:CPU:0'

    if not newline:
        s = s.rstrip()
    return arg

def load_models():
    models = {}
    model = {}
    application = Application()
    app_param_task = application.get_app_param("ImageCaptioning", "ConfigExecutor")
    device = select_device('0' if tf.config.list_physical_devices('GPU') else 'cpu')
    models["device"] = device

    for i in app_param_task:
        key = str(list(i.keys())[0])
        config_device = i[key]['configs']['configDevice']['value']['value']
        if not os.path.exists(weight_caption_path):
            if Download.download_from_drive(weight_url, weight_caption_path) is not None:
                print(f"{'model.h5'} model download successful.")
            else:
                print(f"{'model.h5'} model download failed.")

        weight_path = weight_caption_path
        tf.keras.mixed_precision.set_global_policy('float32')
        model["model"] = tf.keras.models.load_model(weight_path)

        if not os.path.exists(feature_weight_path):
            if Download.download_from_drive(feature_weight_url, feature_weight_path) is not None:
                print(f"{'efficientnetb0_notop.h5'} weights download successful.")
            else:
                print(f"{'efficientnetb0_notop.h5'} weights download failed.")

        efficientnet_path = feature_weight_path
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet')
        base_model.load_weights(efficientnet_path, by_name=True)
        x = GlobalAveragePooling2D()(base_model.output)
        feature_ext_model = Model(inputs=base_model.input, outputs=x)
        models["feature_ext"] = feature_ext_model

        if not os.path.exists(tokenizer_path):
            if Download.download_from_drive(tokenizer_url, tokenizer_path) is not None:
                print("tokenizer download successful.")
            else:
                print("tokenizer download failed.")

        with tf.device(device):
            if config_device == "GPU" and 'GPU' in device:
                tf.keras.mixed_precision.set_global_policy('float32')
                models['ModelGPU'] = model["model"]
            else:
                tf.keras.mixed_precision.set_global_policy('float32')
                models['ModelCPU'] = model["model"]

    return models