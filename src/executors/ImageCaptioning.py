
import os
import sys
import json
import numpy as np
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
    def bootstrap() -> dict:
        model = load_models()
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
        img_temp = img.value
        height, width, _ = img.value.shape
        image_array = img.value
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        features = self.base_model.predict(image_array, verbose=0)
        tokenizer = self.load_tokenizer_from_json()
        caption = self.predict_caption(features, self.model, tokenizer)
        caption = caption.replace('startseq', '').replace('endseq', '').strip()

        if self.concat:
            height, width = img_temp.shape[:2]
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

            dummy_img = PILImage.new("RGB", (width, height))
            dummy_draw = ImageDraw.Draw(dummy_img)

            for word in words:
                test_line = f"{current_line} {word}".strip()
                text_width, _ = dummy_draw.textsize(test_line, font=font)
                if text_width <= max_line_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word

            lines.append(current_line)
            text_height = font.getsize(lines[0])[1]
            line_spacing = 15
            new_height = height + caption_area
            new_img = PILImage.new("RGB", (width, new_height), (255, 255, 255))
            new_img.paste(PILImage.fromarray(img_temp), (0, 0))
            draw = ImageDraw.Draw(new_img)

            y = height + (caption_area - (text_height + line_spacing) * len(lines)) // 2

            for line in lines:
                text_width, _ = draw.textsize(line, font=font)
                text_x = (width - text_width) // 2
                draw.text((text_x, y), line, font=font, fill=font_color)
                y += text_height + line_spacing

            img.value = np.array(new_img)
            return img, caption

    # def caption_infer(self, img):
    #     img_temp = img.value
    #     height, width, _ = img.value.shape
    #     image_array = img.value
    #     image_array = np.expand_dims(image_array, axis=0)
    #     image_array = preprocess_input(image_array)
    #     features = self.base_model.predict(image_array, verbose=0)
    #     tokenizer = self.load_tokenizer_from_json()
    #     caption = self.predict_caption(features, self.model, tokenizer)
    #     caption = caption.replace('startseq', '').replace('endseq', '').strip()
    #
    #     if self.concat:
    #         height, width = img_temp.shape[:2]
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         font_scale = 1
    #         font_color = (0, 0, 0)
    #         line_type = 2
    #         max_line_width = width - 20
    #         words = caption.split(' ')
    #         lines = []
    #         current_line = ""
    #
    #         for word in words:
    #             test_line = f"{current_line} {word}".strip()
    #             (text_width, text_height), _ = cv2.getTextSize(test_line, font, font_scale, line_type)
    #             if text_width <= max_line_width:
    #                 current_line = test_line
    #             else:
    #                 lines.append(current_line)
    #                 current_line = word
    #
    #         lines.append(current_line)
    #         text_height = cv2.getTextSize(lines[0], font, font_scale, line_type)[1]
    #         line_spacing = 15
    #         caption_area = 100
    #         new_height = height + caption_area
    #         new_img = np.zeros((new_height, width, 3), dtype=np.uint8)
    #         new_img[:height, :] = img_temp
    #         new_img[height:] = (255, 255, 255)
    #         y = height + (caption_area - (text_height + line_spacing) * len(lines)) // 2 + text_height
    #
    #         for line in lines:
    #             text_width, _ = cv2.getTextSize(line, font, font_scale, line_type)[0]
    #             text_x = (width - text_width) // 2
    #             cv2.putText(new_img, line, (text_x, y), font, font_scale, font_color, line_type)
    #             y += text_height + line_spacing
    #         img.value = new_img
    #         return img, caption

    def run(self):
        self.image = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if not self.image: return None
        self.image, self.caption = self.caption_infer(self.image)
        self.image = Image.set_frame(img=self.image, package_uID=self.uID, redis_db=self.redis_db)
        packageModel = build_response(context=self)
        return Response(model=packageModel, bootstrap=self.bootstrap).response()


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
