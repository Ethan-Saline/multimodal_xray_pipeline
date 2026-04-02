from health_multimodal.image.utils import ImageModelType, get_image_inference

engine = get_image_inference(ImageModelType.BIOVIL)
# Inspect what's available
print(engine.model)  # look for the final conv layer name