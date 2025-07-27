import os
import platform
import tensorflow as tf
from keras.layers import LSTM,GlobalAveragePooling2D,Bidirectional
from keras.models import Model
from sdks.novavision.src.base.application import Application
from sdks.novavision.src.base.logger import LoggerManager
from sdks.novavision.src.base.download import Download
from keras.models import load_model

loggerManager = LoggerManager()

MODEL_ASSETS = {
    'caption_model': {
        'path': 'storage/model.h5',
        'url': 'https://drive.google.com/file/d/1omwaXt8-Hnq4YLNizl3PFy0OE7EyzoOh/view?usp=sharing',
        'name': 'Caption Model'
    },
    'feature_model_weights': {
        'path': '/storage/efficientnetb0_notop.h5',
        'url': 'https://drive.google.com/file/d/1EeKtWayy7NHqozqeb4-dARX52o-jxq-6/view?usp=sharing',
        'name': 'Feature Extractor Weights'
    },
    'tokenizer': {
        'path': '/storage/tokenizer.json',
        'url': 'https://drive.google.com/file/d/1rym52IjLNjwE--V1nHH65YuqDf6eP4Cl/view?usp=sharing',
        'name': 'Tokenizer'
    }
}

def _download_asset_if_needed(asset_name):
    asset = MODEL_ASSETS[asset_name]
    path = asset['path']
    name = asset['name']
    if not os.path.exists(path):
        loggerManager.info(f"Downloading {name}...")
        if Download.download_from_drive(asset['url'], path):
            loggerManager.info(f"{name} download successful.")
            return True
        else:
            loggerManager.error(f"{name} download failed.")
            raise IOError(f"Failed to download required asset: {name}")
    loggerManager.info(f"{name} already exists.")
    return True

def select_device(device='', batch_size=0, newline=True):
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
    loggerManager.info(s.strip())
    return arg

def load_models(config):
    """
    model = load_model('/storage/model.h5', compile=False)
    model.summary()
    """
    models = {}
    try:
        
        application = Application()
        device_preference = application.get_param(config=config, name="ConfigDevice")
        loggerManager.info(f"Device preference from UI: {device_preference}")



        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        device_to_use = '0' if gpu_available and device_preference == "GPU" else 'cpu'
        device_path = select_device(device_to_use)
        models["device"] = device_path



        _download_asset_if_needed('caption_model')
        _download_asset_if_needed('feature_model_weights')
        _download_asset_if_needed('tokenizer')
        models["tokenizer_path"] = MODEL_ASSETS['tokenizer']['path']
        


        def legacy_lstm_patch(*args, **kwargs):
            kwargs.pop('time_major', None)
            kwargs.pop('implementation', None)
            return LSTM(*args, **kwargs)

        with tf.keras.utils.custom_object_scope({'LSTM': legacy_lstm_patch}):
            caption_model = tf.keras.models.load_model(MODEL_ASSETS['caption_model']['path'])
        
        loggerManager.info("Caption model loaded successfully in compatibility mode.")
                


        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet')
        base_model.load_weights(MODEL_ASSETS['feature_model_weights']['path'], by_name=True)
        x = GlobalAveragePooling2D()(base_model.output)
        models["feature_ext"] = Model(inputs=base_model.input, outputs=x)
        loggerManager.info("Feature extractor model loaded successfully.")



        with tf.device(device_path):
            tf.keras.mixed_precision.set_global_policy('float32')
            if device_preference == "GPU" and 'GPU' in device_path:
                models['ModelGPU'] = caption_model
                loggerManager.info("Caption model assigned to GPU.")
            else:
                models['ModelCPU'] = caption_model
                loggerManager.info("Caption model assigned to CPU.")

        loggerManager.info("All models loaded and configured successfully!")
        return models

    except Exception as e:
        loggerManager.error(f"A critical error occurred during model loading: {e}")
        import traceback
        loggerManager.error(f"Traceback: {traceback.format_exc()}")
        return {}
