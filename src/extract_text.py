import cv2
from TextSegmentation.text_segmentation import segmentImage
from Image2Text.eval import evaluate

image_path = "../data/sample/a00-000u.png"
image = cv2.imread(image_path)

segments = segmentImage(image)
print(evaluate(segments))


