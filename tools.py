from langchain.tools import BaseTool
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection

def get_image_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# source transformer page in huggingface
def detect_objects(image_path):
    image = Image.open(image_path).convert('RGB')

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # only keeping detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detections = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        detections += ' {}'.format(model.config.id2label[int(label)])
        detections += ' {}\n'.format(float(score))

    return detections

class ImageCaptionTool(BaseTool):
  name: str = "Image captioner"
  # this is how the agent will know what caption to use from all the others tools
  description: str = "Use this tool when given the path of an image that u would like to described. Tt will return a simple caption describing the image."

  def _run(self, image_path: str):
    return get_image_caption(image_path)

  def _arun(self, query: str):
    raise NotImplementedError("This tool does not support async")

class ObjectDetectionTool(BaseTool):
  name: str="Object detector"
  description: str="Use this tool when given the path of an image that you would like to detect object. It will return a list of all detected objects. Each element in the list is in the format: [x1, y1, x2, y2] class_name confidence_score "

  def _run(self, image_path: str):
    return detect_objects(image_path)

  def _arun(self, query: str):
    raise NotImplementedError("This tool does not support async")