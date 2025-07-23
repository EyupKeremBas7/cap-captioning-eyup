
import os
import sys
import cv2
import json
import base64
import requests

sys.path.append(os.path.join(os.path.dirname(__file__),'../../../'))

from sdks.novavision.src.base.model import Image
from capsules.ImageCaptioning.src.models.PackageModel import PackageModel,PackageConfigs,ImageCaptioningConfigs,ImageCaptioningInputs,ImageCaptioningExecutor,ImageCaptioningRequest,InputImage,ConfigExecutor, ConfigTemperature,ConfigConcatenatedCaption, ConfigConcatenatedCaptionTrue, ConfigDevice, ConfigDeviceCPU, ConfigDeviceGPU


ENDPOINT_URL = "http://127.0.0.1:8000/api"


def inference():
    image = cv2.imread("/opt/project/capsules/ImageCaptioning/resources/bicycle.jpg")
    mimeType = "image/jpg"
    mimeType = mimeType.split("/")
    bin = cv2.imencode("." + str(mimeType[1]), image)[1]
    data = str(base64.b64encode(bin), "utf-8")
    image_data = Image(name="image", uID="Oeh5oF", mimeType="image/jpg", encoding="base64", value=data, type="Image")
    concatenateTrue = ConfigConcatenatedCaptionTrue(value=True)
    concatenate = ConfigConcatenatedCaption(value=concatenateTrue)
    temperature = ConfigTemperature(value=0.5)
    configDeviceGPU = ConfigDeviceGPU(value="GPU")
    configDeviceCPU = ConfigDeviceCPU(value="CPU")
    configDevice = ConfigDevice(value=configDeviceCPU)
    captionConfigs = ImageCaptioningConfigs(configTemperature=temperature, configConcatenatedCaption=concatenate, configDevice=configDevice)
    inputImage = InputImage(value=image_data)
    captionInputs = ImageCaptioningInputs(inputImage=inputImage)
    captionRequest = ImageCaptioningRequest(inputs=captionInputs, configs=captionConfigs)
    captionExecutor = ImageCaptioningExecutor(value=captionRequest)
    executor = ConfigExecutor(value=captionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    request = PackageModel(configs=packageConfigs)
    request_json = json.loads(request.json())
    response = requests.post(ENDPOINT_URL, json=request_json)
    print(response.raise_for_status())
    print(response.json())


if __name__ =="__main__":
    inference()