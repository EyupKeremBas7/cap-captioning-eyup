from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Inputs, Configs, Outputs, Response, Request, Output, Input, Config, Image


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image],Image]
    type: str  = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"
    class Config:
        title="Image"



class OutputCaption(Output):
    name: Literal["outputCaption"] = "outputCaption"
    value: str
    type: Literal["string"] = "string"

    class Config:
        title = "Caption"


class ImageCaptioningInputs(Inputs):
    inputImage: InputImage


class ConfigTemperature(Config):
    """
    Temperature variable is used to control the randomness of the predictions during decoding. Lower temperatures make the model's predictions more deterministic, while higher temperatures increase diversity and randomness in the generated captions.
    """
    name: Literal["Temperature"] = "Temperature"
    value: float = Field(default=0.5, ge=0.2, le=1.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Temperature"

class ConfigDeviceGPU(Config):
    name: Literal["ConfigDeviceGPU"] = "ConfigDeviceGPU"
    value: Literal["GPU"] = "GPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "GPU"


class ConfigDeviceCPU(Config):
    name: Literal["ConfigDeviceCPU"] = "ConfigDeviceCPU"
    value: Literal["CPU"] = "CPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "CPU"


class ConfigDevice(Config):
    """
        It refers to whether the model should run on a CPU or a GPU.
        You can select the device type for inference or training process.
    """
    name: Literal["ConfigDevice"] = "ConfigDevice"
    value: Union[ConfigDeviceCPU, ConfigDeviceGPU]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Device"


class ImageCaptioningConfigs(Configs):
    configDevice: ConfigDevice
    configTemperature: ConfigTemperature


class ImageCaptioningOutputs(Outputs):
    outputCaption: OutputCaption


class ImageCaptioningRequest(Request):
    inputs: Optional[ImageCaptioningInputs]
    configs: ImageCaptioningConfigs
    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class ImageCaptioningResponse(Response):
    outputs: ImageCaptioningOutputs


class ImageCaptioningExecutor(Config):
    name: Literal["ImageCaptioning"] = "ImageCaptioning"
    value: Union[ImageCaptioningRequest, ImageCaptioningResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Image Captioning"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[ImageCaptioningExecutor]
    type:Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target" : "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["ImageCaptioning"] = "ImageCaptioning"
