
from sdks.novavision.src.helper.package import PackageHelper
from capsules.ImageCaptioning.src.models.PackageModel import ImageCaptioningExecutor, PackageModel, PackageConfigs, ImageCaptioningResponse, ImageCaptioningOutputs, OutputImage, ConfigExecutor, OutputCaption

def build_response(context):
    outputCaption = OutputCaption(value=context.caption)
    captionOutputs = ImageCaptioningOutputs(outputImage=outputImage, outputCaption=outputCaption)
    captionResponse = ImageCaptioningResponse(outputs=captionOutputs)
    captionExecutor = ImageCaptioningExecutor(value=captionResponse)
    executor = ConfigExecutor(value=captionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel