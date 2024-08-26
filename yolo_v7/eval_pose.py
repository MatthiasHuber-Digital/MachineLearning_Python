import cv2
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms

from utils.datasets import letterbox
from models.experimental import attempt_load 
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint
from utils.pose import show_image, plot_pose_prediction, make_pose_prediction, add_pose_in_video_file, scale_pose_output
from utils.general import non_max_suppression_kpt, xywh2xyxy
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box


# read original image
orig_img = cv2.imread('inference/images/image1.jpg')
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
original_shape = orig_img.shape
print('Original image shape: ', original_shape)
cv2.imwrite("original_test_image.jpg", orig_img)

# resize image without padding
img = cv2.resize(orig_img, (640, 640), interpolation=cv2.INTER_LINEAR)
print('Resized image', img.shape)
cv2.imwrite("reshaped_test_image.jpg", img)

#print('Anchors:', model.yaml['anchors'])
image_tensor = transforms.ToTensor()(img)
image_tensor = torch.unsqueeze(image_tensor, 0)
#print('Transformed to tensor image:', image_tensor.shape)
image_tensor = image_tensor.to(device).float()
with torch.no_grad():
    pred, _ = model(image_tensor)
print('Predictions shape:', pred.shape)

predicted_keypoints = non_max_suppression_kpt(pred, 
                               conf_thres=0.25, 
                               iou_thres=0.65, 
                               nc=model.yaml['nc'], 
                               nkpt=model.yaml['nkpt'], 
                               kpt_label=True)
print('Detected poses:', len(predicted_keypoints))
print('Prediction shape:', predicted_keypoints[0].shape)


predicted_keypoints = output_to_keypoint(predicted_keypoints)

def plot_pose_prediction(
        img: cv2.Mat,
        pred: list, 
        line_thickness: int = 2, 
        show_bbox: bool = False,
    ) -> cv2.Mat:
    bbox = xywh2xyxy(pred[:,2:6])
    for idx in range(pred.shape[0]):
        plot_skeleton_kpts(img, pred[idx, 7:].T, 3)
        if show_bbox:
            plot_one_box(bbox[idx], img, line_thickness=line_thickness)

plot_pose_prediction(img, predicted_keypoints)
cv2.imwrite("detections_test_image.jpg", img)

scaled_predicted_keypoints = scale_pose_output(
    scaled_keypt_output=predicted_keypoints,
    original_shape=original_shape,
    resized_shape=(640, 640),
    is_padded=False,
)
plot_pose_prediction(orig_img, scaled_predicted_keypoints)
cv2.imwrite("detections_test_image.jpg", orig_img)

#add_pose_in_video_file('board.mp4', 'board_out.mp4')

def pose_prediction_single_image(
        image: cv2.Mat
):
    plot_pose_prediction(img, predicted_keypoints)
    cv2.imwrite("detections_test_image.jpg", img)

    scaled_predicted_keypoints = scale_pose_output(
        scaled_keypt_output=predicted_keypoints,
        original_shape=original_shape,
        resized_shape=(640, 640), # natural shape for yolov7 images
        is_padded=False,
    )
    plot_pose_prediction(orig_img, scaled_predicted_keypoints)
    cv2.imwrite("detections_test_image.jpg", orig_img)


def pose_prediction_dataset(opt: dict):
        
    os.chdir('MachineLearning_Python/yolo_v7')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    model = attempt_load('weights_pretrained_models/yolov7_w6_pretrained_pose.pt', map_location=device)
    model.eval()

    print('Number of classes:', model.yaml['nc'])
    print('Number of keypoints:', model.yaml['nkpt'])


    for image in list_images:
        pose_prediction_single_image(
            image=preprocessed_image,
        )



if __name__ == '__main__':

    import sys
    import datetime
    old_stdout = sys.stdout

    time_and_date = str(datetime.datetime.now())
    log_file_name = "MachineLearning_Python/yolo_v7/runs/train/training_yolov7_" + time_and_date + ".log"
    log_file = open(log_file_name,"w")

    sys.stdout = log_file

    device = 'cuda:0' if torch.cuda.is_available else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='MachineLearning_Python/yolo_v7/weights_pretrained_models/yolov7_w6_pretrained_COCO.pt', help='initial weights path')
    parser.add_argument('--device', type=str, default=device, help='computation device')
    parser.add_argument('--data', type=str, default='ds_hard_hat/data.yaml', help='data.yaml path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    
    pose_prediction_dataset(opt)

    sys.stdout = old_stdout

    log_file.close()