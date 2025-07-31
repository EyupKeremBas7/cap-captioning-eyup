
import os
import sys
import json
import numpy as np
import tensorflow as tf
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.response import Response
from sdks.novavision.src.helper.executor import Executor
from capsules.ImageCaptioning.src.utils.utils import load_models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from capsules.ImageCaptioning.src.utils.response import build_response
from tensorflow.keras.applications.efficientnet import preprocess_input
from capsules.ImageCaptioning.src.models.PackageModel import PackageModel


class ImageCaptioning(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.select_device = self.bootstrap["device"]
        self.base_model = self.bootstrap["feature_ext"]
        self.image = self.request.get_param("inputImage")
        self.device = self.request.get_param("ConfigDevice")
        self.temperature = self.request.get_param("Temperature")
        self.concat = self.request.get_param("ConcatenatedCaption")
        if self.device == "GPU" and "GPU" in self.select_device:
            self.model = self.bootstrap["ModelGPU"]
        else:
            self.model = self.bootstrap["ModelCPU"]

    @staticmethod
    def bootstrap(config:dict) -> dict:
        model = load_models(config)
        return model

    def load_tokenizer_from_json(self):
        with open("/storage/tokenizer.json", "r", encoding="utf-8") as f:
            tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)
        return tokenizer

    def word_for_id(self, integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def predict_caption(self, image, model, tokenizer, max_length=34):
        in_text = 'startseq'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
            yhat = model.predict([image, sequence], verbose=0)
            yhat = np.asarray(yhat).astype('float64')
            yhat = np.log(yhat + 1e-10) / self.temperature
            exp_preds = np.exp(yhat)
            yhat = exp_preds / np.sum(exp_preds)
            yhat = yhat.flatten()
            yhat = np.random.choice(range(len(yhat)), p=yhat)
            word = self.word_for_id(yhat, tokenizer)
            if word is None:
                break
            in_text += " " + word
            if word == 'endseq':
                break
        return in_text

    def caption_infer(self, img):
        if hasattr(img, 'value'):
            img_temp = img.value
        else:
            img_temp = img
        if not isinstance(img_temp, np.ndarray):
            img_temp = np.array(img_temp)
        if img_temp.dtype == np.float32 or img_temp.dtype == np.float64:
            if img_temp.max() <= 1.0:
                img_temp = (img_temp * 255).astype(np.uint8)
            else:
                img_temp = img_temp.astype(np.uint8)
        elif img_temp.dtype != np.uint8:
            img_temp = img_temp.astype(np.uint8)
        
        if len(img_temp.shape) == 2:
            img_temp = np.stack([img_temp] * 3, axis=-1)
        elif len(img_temp.shape) == 3:
            if img_temp.shape[2] == 1:
                img_temp = np.repeat(img_temp, 3, axis=2)
            elif img_temp.shape[0] == 1 and img_temp.shape[1] == 1:
                img_temp = np.repeat(np.repeat(img_temp, 224, axis=0), 224, axis=1)
        
        if img_temp.shape[0] < 10 or img_temp.shape[1] < 10:
            from scipy.ndimage import zoom
            scale_h = max(1, 224 / img_temp.shape[0])
            scale_w = max(1, 224 / img_temp.shape[1])
            img_temp = zoom(img_temp, (scale_h, scale_w, 1), order=1).astype(np.uint8)
        
        height, width = img_temp.shape[:2]
        image_resized = tf.image.resize(img_temp, [224, 224])
        
        image_array = tf.expand_dims(image_resized, axis=0)
        image_array = preprocess_input(image_array)
        
        features = self.base_model.predict(image_array, verbose=0)
        tokenizer = self.load_tokenizer_from_json()
        caption = self.predict_caption(features, self.model, tokenizer)
        caption = caption.replace('startseq', '').replace('endseq', '').strip()

        if self.concat:
            font_size = 24
            font_color = (0, 0, 0)
            caption_area = 100
            max_line_width = width - 20

            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

            words = caption.split(' ')
            lines = []
            current_line = ""

            dummy_img = PILImage.new("RGB", (max(width, 100), max(height, 100)))
            dummy_draw = ImageDraw.Draw(dummy_img)

            for word in words:
                test_line = f"{current_line} {word}".strip()
                
                bbox = dummy_draw.textbbox((0, 0), test_line, font=font)
                text_width = bbox[2] - bbox[0]
                
                if text_width <= max_line_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word

            if current_line: 
                lines.append(current_line)
            
            if lines:
                bbox = dummy_draw.textbbox((0, 0), lines[0], font=font)
                text_height = bbox[3] - bbox[1]
                
                line_spacing = 15
                new_height = height + caption_area
                new_img = PILImage.new("RGB", (width, new_height), (255, 255, 255))
                

                pil_img = PILImage.fromarray(img_temp)
                new_img.paste(pil_img, (0, 0))
                draw = ImageDraw.Draw(new_img)

                y = height + (caption_area - (text_height + line_spacing) * len(lines)) // 2

                for line in lines:
                   
                    bbox = draw.textbbox((0, 0), line, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_x = max(0, (width - text_width) // 2)
                    draw.text((text_x, y), line, font=font, fill=font_color)
                    y += text_height + line_spacing

                img.value = np.array(new_img)
            else:
                img.value = img_temp
        else:
            img.value = img_temp
        return img, caption


    def run(self):
        self.image = Image.get_frame(img=self.image, redis_db=self.redis_db)
        self.image, self.caption = self.caption_infer(self.image)
        print(self.caption)
        self.image = Image.set_frame(img=self.image, package_uID=self.uID, redis_db=self.redis_db)
        packageModel = build_response(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
