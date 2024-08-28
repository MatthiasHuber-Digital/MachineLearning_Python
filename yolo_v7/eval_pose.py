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
from utils.pose import show_image, plot_pose_prediction, make_pose_prediction, add_pose_in_video_file, scale_pose_output
from utils.general import non_max_suppression_kpt, xywh2xyxy
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box


def split_off_file_extension(file_path: str) -> tuple:
    """This function splits off the file extension off an entire file path.

    Args:
        file_path (str): The file path including the file extension.

    Returns:
        file_path_no_extension (str): The file path without the LAST file extension.
        file_extension (str): The LAST file extension as separate string.
    """
    list_pieces = file_path.split('.')
    file_extension = list_pieces[-1]
    file_path_no_extension = '.'.join(list_pieces[:-1])

    return file_path_no_extension, file_extension


def load_dataset(
    path_dataset: str,
    image_file_extension: str = 'jpg',
) -> list:
    """This function returns a list of tuples which contain
    both image data and image path.

    Args:
        path_dataset (str): The dataset folder path.
        image_file_extension (str): File extension of the image files to work on.
    Returns:
        list_images_and_paths (list): List of tuples of image data and path of images.
    """
    list_image_paths = list(pathlib.Path(path_dataset).glob('*.' + image_file_extension))
    list_image_paths = [str(path) for path in list_image_paths]
    
    list_images = [None] * len(list_image_paths)
    for idx, path in enumerate(list_image_paths):
        list_images[idx] = cv2.imread(path)

    list_images_and_paths = list(zip(list_images, list_image_paths))

    return list_images_and_paths


def pose_prediction_single_image(
    orig_img: cv2.Mat,
    img_path_wo_ext: str,
    img_ext: str,
    device: torch.device,
    model,
):
    """This function conducts a pose prediction on a single image, inlcuding saving the image files.

    Args:
        orig_img (cv2.Mat): Opencv-data of the original image.
        img_path_wo_ext (str): Path of the original image without file extension.
        img_ext (str): Image file extension.
        model (_type_): Prediction model (yolo v7).
    """
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(img_path_wo_ext + '_postproc.' + img_ext, img)

    image_tensor = transforms.ToTensor()(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    image_tensor = image_tensor.to(device).float()

    with torch.no_grad():
        pred, _ = model(image_tensor)

    predicted_keypoints = non_max_suppression_kpt(pred, 
                                conf_thres=0.25, 
                                iou_thres=0.65, 
                                nc=model.yaml['nc'], 
                                nkpt=model.yaml['nkpt'], 
                                kpt_label=True)

    predicted_keypoints = output_to_keypoint(predicted_keypoints)

    plot_pose_prediction(img, predicted_keypoints)
    
    cv2.imwrite(img_path_wo_ext + '_scaled_with_poses.' + img_ext, img)

    scaled_predicted_keypoints = scale_pose_output(
        scaled_keypt_output=predicted_keypoints,
        original_shape=orig_img.shape,
        resized_shape=(640, 640),
        is_padded=False,
    )
    plot_pose_prediction(orig_img, scaled_predicted_keypoints)
    cv2.imwrite(img_path_wo_ext + '_orig_scale_with_poses.' + img_ext, orig_img)


def pose_prediction_dataset(
    path_dataset: str,
    path_model: str,
    path_workdir: str = None,
):
    """This function predicts the poses for all images in a dataset.

    Args:
        path_dataset (str): Path of the dataset folder.
        path_model (str): Path to the model including weights (yolov7 with pose prediction).
        path_workdir (str): Path to the workdir to which to switch to before executing the funciton.
    """
    if path_workdir is not None:
        os.chdir(path_workdir)
    
    list_images_and_paths = load_dataset(path_dataset=path_dataset, image_file_extension='jpg')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = attempt_load(path_model, map_location=device)
    model.eval()

    for img_data, img_path in list_images_and_paths:
        path_wo_ext, ext = split_off_file_extension(file_path=img_path)

        pose_prediction_single_image(
            orig_img=img_data,
            img_path_wo_ext=path_wo_ext,
            img_ext=ext,
            device=device,
            model=model,
        )


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
    
    pose_prediction_dataset(
        path_dataset=opt.data, 
        path_model=opt.weights,
        path_workdir='MachineLearning_Python/yolo_v7')

    sys.stdout = old_stdout

    log_file.close()