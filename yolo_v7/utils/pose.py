import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, xywh2xyxy
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box


def scale_pose_output(
            scaled_keypt_output, 
            resized_shape: tuple,
            original_shape: tuple,
            is_padded: bool=True,
        ):
    ''' Scale yolo pose estimator output coordinates of bbox and keypoints
    from `resized_shape` to `original_shape` 
    '''
    scaled_output = scaled_keypt_output.copy()
    scale_ratio = (resized_shape[1] / original_shape[1], 
                   resized_shape[0] / original_shape[0])      
    if is_padded:
        # remove padding
        pad_scale = min(scale_ratio)
        padding = (resized_shape[1] - original_shape[1] * pad_scale ) / 2, (
                   resized_shape[0] - original_shape[0] * pad_scale ) / 2
        scale_ratio = (pad_scale, pad_scale)
        
        scaled_output[:, 2] -= padding[0]     # x_c unpadding
        scaled_output[:, 3] -= padding[1]     # y_c unpadding
        scaled_output[:, 7::3] -= padding[0]  # x_kpts unpadding
        scaled_output[:, 8::3] -= padding[1]  # y_kpts unpadding
    
    scaled_output[:, [2, 4]] /= scale_ratio[0]
    scaled_output[:, [3, 5]] /= scale_ratio[1]
    scaled_output[:, 7::3] /= scale_ratio[0]
    scaled_output[:, 8::3] /= scale_ratio[1]

    return scaled_output


def plot_pose_prediction(img: cv2.Mat, 
                         pred: list, thickness=2, 
                         show_bbox: bool=True,
                         ) -> cv2.Mat:
    bbox = xywh2xyxy(pred[:,2:6])
    for idx in range(pred.shape[0]):
        plot_skeleton_kpts(img, pred[idx, 7:].T, 3)
        if show_bbox:
            plot_one_box(bbox[idx], img, line_thickness=thickness)


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


def pose_prediction_image_dataset(
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


def make_pose_prediction(
            model, 
            img: cv2.Mat,
            device: str,
        ) -> list:
    # Resize and pad image while meeting stride-multiple constraints
    img_ = letterbox(img, 960, stride=64, auto=True)[0]
    resized_shape = img_.shape[0:2]
    # Transform image to model readable structure
    img_ = transforms.ToTensor()(img_)
    img_ = torch.tensor(np.array([img_.numpy()]))
    img_ = img_.to(device).float()
    with torch.no_grad():
        output, _ = model(img_)
    # Filter predictions
    output = non_max_suppression_kpt(output, 0.25, 0.65, 
                                     nc=model.yaml['nc'], 
                                     nkpt=model.yaml['nkpt'], 
                                     kpt_label=True)
    output = output_to_keypoint(output)
    # scale to original image shape
    output = scale_pose_output(output, resized_shape, img.shape[0:2])
    return output


def add_pose_in_video_file(
        model,
        path_video_file: str,
        ):
    """This function adds the pose into a video file of mp4 format.

    Args:
        model (pt): Trained yolov7 pose model file (.pt).
        path_video_file (str): Path to the video file (mp4 only).
    """
    # Open the input video file and extract its properties
    cap = cv2.VideoCapture(path_video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # Create VideoWriter object 
    out = cv2.VideoWriter(path_video_file, fourcc, fps, (width, height))
    #  Processing a video file frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            pred = make_pose_prediction(model, frame)
            plot_pose_prediction(frame, pred, show_bbox=False)
            out.write(frame)
            cv2.imshow('Pose estimation', frame)
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    # Release VideoCapture and VideoWriter
    cap.release()
    out.release()
    # Close all frames and video windows
    cv2.destroyAllWindows()


def pose_prediction_video_dataset(
    path_dataset: str,
    path_model: str,
    path_workdir: str = None,
):
    """This function predicts the poses for all videos in a dataset.

    Args:
        path_dataset (str): Path of the dataset folder.
        path_model (str): Path to the model including weights (yolov7 with pose prediction).
        path_workdir (str): Path to the workdir to which to switch to before executing the funciton.
    """
    