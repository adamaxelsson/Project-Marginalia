import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os

current_folder_path = "/cephyr/users/adamax/Alvis/Project-Marginalia/model-to-words/"
img_folder_path = "/cephyr/users/adamax/Alvis/Project-Marginalia/model-to-words/original_images/"
output_folder_path = "/cephyr/users/adamax/Alvis/Project-Marginalia/model-to-words/marginalia/"
model_path = "/cephyr/users/adamax/Alvis/Project-Marginalia/model-to-words/model.pt"
model_input_height = 350
model_input_width = 500

def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    num_classes = 2  # 1 class (marginalia) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(original_img):
    processed_img = cv2.resize(original_img, (model_input_width, model_input_height))
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    processed_img = processed_img/255
    processed_img = np.dstack([processed_img])
    processed_img = torch.tensor(processed_img, dtype=torch.float32)
    processed_img = processed_img.permute(2,0,1) # change channel position
    processed_img = [processed_img.to(device)]
    return processed_img

def predict_bboxes(model, image):
    pred = model(image)
    boxes = pred[0]['boxes']
    scores = pred[0]['scores']
    detection_threshold = 0.05
    keep = torchvision.ops.nms(boxes, scores, detection_threshold)
    tensor_bounding_box = boxes.cpu().detach().numpy()
    return tensor_bounding_box

images = os.listdir(img_folder_path)
model = load_model()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


for i, image_name in enumerate(images):
    original_img = cv2.imread(img_folder_path + image_name) 
    height, width, channels = original_img.shape
    ratio_height = height / model_input_height
    ratio_width = width / model_input_width
    processed_image = preprocess_image(original_img)
    bboxes = predict_bboxes(model, processed_image)

    #save bboxes
    for j, box in enumerate(bboxes):
        x_min = box[0] * ratio_width
        y_min = box[1] * ratio_height
        x_max = box[2] * ratio_width
        y_max = box[3] * ratio_height

        cropped = original_img[int(y_min):int(y_max), int(x_min):int(x_max)]
        cv2.imwrite(output_folder_path+image_name[:-4]+"_"+str(j)+".png", cropped)
