from matplotlib import pyplot as plt
from yolov7.utils.general import xywh2xyxy
from yolov7.utils.plots import plot_skeleton_kpts, plot_one_box
import cv2


def show_image(img, figsize=(6, 6)):            
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def plot_pose_prediction(img: cv2.Mat, 
                         pred: list, thickness=2, 
                         show_bbox: bool=True,
                         ) -> cv2.Mat:
    bbox = xywh2xyxy(pred[:,2:6])
    for idx in range(pred.shape[0]):
        plot_skeleton_kpts(img, pred[idx, 7:].T, 3)
        if show_bbox:
            plot_one_box(bbox[idx], img, line_thickness=thickness)
