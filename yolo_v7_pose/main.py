import requests
WEIGHTS_URL = 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt'
open("weights/yolov7-w6-pose.pt", "wb").write(requests.get(WEIGHTS_URL).content)

import cv2
import torch
from torchvision import transforms

from yolov7.models.experimental import attempt_load 
from yolov7.utils.datasets import letterbox 
from yolov7.utils.general import non_max_suppression_kpt
from yolov7.utils.plots import output_to_keypoint

from own_utils.plots import show_image, plot_pose_prediction
from own_utils.pose import make_pose_prediction, add_pose_in_video_file

# set gpu device if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
# Load model
model = attempt_load('weights/yolov7-w6-pose.pt', map_location=device) 
# Switch to evaluation mode, map_location=device
model.eval()

print('Number of classes:', model.yaml['nc'])
print('Number of keypoints:', model.yaml['nkpt'])


# read original image
orig_img = cv2.imread('test_img.jpg')
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
print('Original image', orig_img.shape)
show_image(orig_img)

# resize and pad
img = letterbox(orig_img, 640, stride=64, auto=True)[0]
print('Resized image', img.shape)
show_image(img)

print('Anchors:', model.yaml['anchors'])

# transform to tensor
image_tensor = transforms.ToTensor()(img)
# add dimension
image_tensor = torch.unsqueeze(image_tensor, 0)
print('Transformed to tensor image:', image_tensor.shape)
# send the picture to the calculating device
image_tensor = image_tensor.to(device).float()

with torch.no_grad():
    pred, _ = model(image_tensor)
print('Predictions shape:', pred.shape)

pred = non_max_suppression_kpt(pred, 
                               conf_thres=0.25, 
                               iou_thres=0.65, 
                               nc=model.yaml['nc'], 
                               nkpt=model.yaml['nkpt'], 
                               kpt_label=True)
print('Detected poses:', len(pred))
print('Prediction shape:', pred[0].shape)


pred = output_to_keypoint(pred)
plot_pose_prediction(img, pred)
show_image(img)

img = cv2.cvtColor(cv2.imread('breakdance.jpg'), cv2.COLOR_BGR2RGB)
pred = make_pose_prediction(model, img)
plot_pose_prediction(img, pred, show_bbox=False)
show_image(img, (18,18))


add_pose_in_video_file('board.mp4', 'board_out.mp4')