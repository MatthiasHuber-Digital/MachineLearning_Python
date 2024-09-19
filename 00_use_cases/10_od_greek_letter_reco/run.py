from simple_detector.detector import Detector
from matplotlib import pyplot as plt
import cv2

detector = Detector()
img = cv2.imread('letters.png')
#img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
bboxes = detector.predict(img)

img_w_boxes = detector.visualize_bboxes(
    img=img,
    bboxes=bboxes,
)

is_written = cv2.imwrite("letters_pred_test.png", img_w_boxes)

if is_written:
    print("Finished")
else:
    print("Error saving.")
#plt.imshow(img_w_boxes)
