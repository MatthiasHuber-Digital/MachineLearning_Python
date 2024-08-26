# Yolov7 transfer learning - hard hat dataset (200 samples / 70-20-10)
# The yoloV7 repo is located here: https://github.com/WongKinYiu/yolov7

#!pip install -r requirements.txt # Install the necessary packages this way

# # Begin Custom Training
# Details on settings changes can be found under: 
# https://blog.roboflow.com/yolov7-custom-dataset-training-tutorial/
import os
os.getcwd()

# download a pretrained model
#get_ipython().system('wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt')

# Check if graphic card works
import torch
torch.cuda.is_available()

# call training via this command...
#!python train.py --batch 8 --epochs 10 --data 'ds_hard_hat/data.yaml' --weights 'MachineLearning_Python/yolo_v7/yolov7_training.pt'
          
# # Evaluation
python detect.py --weights 'runs/train/exp/weights/best.pt' --conf 0.1 --source 'ds_hard_hat/test/images'

import glob
from IPython.display import Image, display

i = 0
limit = 10000 # max images to print
for imageName in glob.glob('/content/yolov7/runs/detect/exp/*.jpg'): #assuming JPG
    if i < limit:
      display(Image(filename=imageName))
      print("\n")
    i = i + 1

# # Reparameterize for Inference
# https://github.com/WongKinYiu/yolov7/blob/main/tools/reparameterization.ipynb

# ## Deploy Model on Roboflow
# 
# Once you have finished training your YOLOv7 model, you’ll have a set of trained weights ready for use. These weights will be in the `/content/runs/train/exp/weights/best.pt` folder of your project. You can upload your model weights to Roboflow Deploy to use your trained weights on our infinitely scalable infrastructure.

# # Deploy Your Model to the Edge
# 
# In addition to using the Roboflow hosted API for deployment, you can use [Roboflow Inference](https://inference.roboflow.com), an open source inference solution that has powered millions of API calls in production environments. Inference works with CPU and GPU, giving you immediate access to a range of devices, from the NVIDIA Jetson to TRT-compatible devices to ARM CPU devices.
# 
# With Roboflow Inference, you can self-host and deploy your model on-device. You can deploy applications using the [Inference Docker containers](https://inference.roboflow.com/quickstart/docker/) or the pip package.
# 
# For example, to install Inference on a device with an NVIDIA GPU, we can use:
# 
# ```
# docker pull roboflow/roboflow-inference-server-gpu
# ```
# 
# Then we can run inference via HTTP:
# 
# ```python
# import requests
# 
# workspace_id = ""
# model_id = ""
# image_url = ""
# confidence = 0.75
# api_key = ""
# iou_threshold = 0.5
# 
# infer_payload = {
#     "image": {
#         "type": "url",
#         "value": image_url,
#     },ep
#     "confidence": confidence,
#     "iou_threshold": iou_threshold,
#     "api_key": api_key,
# }
# res = requests.post(
#     f"http://localhost:9001/{workspace_id}/{model_id}",
#     json=infer_object_detection_payload,
# )
# 
# predictions = res.json()
# ```
# 
# Above, set your Roboflow workspace ID, model ID, and API key.
# 
# - [Find your workspace and model ID](https://docs.roboflow.com/api-reference/workspace-and-project-ids?ref=blog.roboflow.com)
# - [Find your API key](https://docs.roboflow.com/api-reference/authentication?ref=blog.roboflow.com#retrieve-an-api-key)
# 
# Also, set the URL of an image on which you want to run inference. This can be a local file.
# 
# _To use your YOLOv5 model commercially with Inference, you will need a Roboflow Enterprise license, through which you gain a pass-through license for using YOLOv5. An enterprise license also grants you access to features like advanced device management, multi-model containers, auto-batch inference, and more._

# # Next sts
# 
# Congratulations, you've trained a custom YOLOv7 model! Next, start thinking about deploying and [building an MLOps pipeline](https://docs.roboflow.com) so your model gets better the more data it sees in the wild.
