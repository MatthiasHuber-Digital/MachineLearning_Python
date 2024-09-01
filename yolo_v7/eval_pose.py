import argparse
import cv2
import matplotlib.pyplot as plt
import os
import pathlib
import torch
from torchvision import transforms

from utils.datasets import letterbox
from models.experimental import attempt_load 
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint
from utils.pose import plot_pose_prediction, make_pose_prediction, add_pose_in_video_file, scale_pose_output
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint


if __name__ == '__main__':

    import sys
    import datetime
    old_stdout = sys.stdout

    time_and_date = str(datetime.datetime.now())
    log_file_name = "MachineLearning_Python/yolo_v7/runs/detect/training_yolov7_" + time_and_date + ".log"
    log_file = open(log_file_name,"w")

    sys.stdout = log_file

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights_pretrained_models/yolov7_w6_pretrained_pose.pt', help='initial weights path')
    parser.add_argument('--data', type=str, default='ds_hard_hat/test/images', help='path to the folder containing the images which to make the pose prediction for')
    opt = parser.parse_args()
    
    if opt.data is not None:
        pose_prediction_image_dataset(
            path_dataset=opt.data,
            path_model=opt.weights,
            path_workdir='MachineLearning_Python/yolo_v7')
    
        pose_prediction_video_dataset(
            path_dataset=opt.data,
            path_model=opt.weights,
            path_workdir='MachineLearning_Python/yolo_v7')
    else:
        raise ValueError("No data path was given by the user.")

    sys.stdout = old_stdout

    log_file.close()
