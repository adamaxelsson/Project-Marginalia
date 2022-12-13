from __future__ import division
import torch
import torchvision
import pandas as pd
import os
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T


class MarginaliaDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.n_samples = len(data)

    def __getitem__(self, index):
        img = self.data[index]["data"]
        boxes = self.data[index]["boxes"]
        labels = self.data[index]["labels"]
        id = self.data[index]["image_id"]
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        return img, target, id
    
    def __len__(self):
        return self.n_samples


def preprocessing(imageID, path):
    """reads in image and returns preprocessed np array"""
    img = cv.imread(f"{path}{imageID}.png")
    img = img/255
    img = torch.tensor(img, dtype=torch.float32)
    img = img.permute(2,0,1) # change channel position
    return img


def generate_data(image_list, box_df, image_path="./data/rescaled_png_files/"):
    data = []
    all_box_coordinates = []
    for image in image_list:
        image_dict = {}
        if image.endswith(".png"): 
            id = image[:-4]
        else: 
            continue

        sub_df = box_df[box_df["number"] == int(id)]
        num_boxes = len(sub_df)
        box_coordinates = []
        for i in range(num_boxes):
            sub_sub_df = sub_df.iloc[i]
            xmin_scaled = int(sub_sub_df["xmin_scaled"])
            ymin_scaled = int(sub_sub_df["ymin_scaled"])
            xmax_scaled = int(sub_sub_df["xmax_scaled"])
            ymax_scaled = int(sub_sub_df["ymax_scaled"])
            box_coordinates.append(torch.tensor([xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled], dtype=torch.int32))
        if num_boxes > 1:
            box_coordinates = torch.stack(box_coordinates, axis=0)
        elif num_boxes == 1:
            box_coordinates = box_coordinates[0]
            box_coordinates = box_coordinates.view(1,4)
        else:
            pass
        all_box_coordinates.append(box_coordinates)
        
        image_data = preprocessing(id, image_path) # returns list

        # labels
        labels = torch.ones(num_boxes, dtype=torch.int64)

        # stack it to dict
        image_dict["data"] = image_data
        image_dict["boxes"] = box_coordinates
        image_dict["labels"] = labels
        image_dict["image_id"] = id

        data.append(image_dict)
    return data



def bbox_iou(boxA, boxB):
  # from https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4
  # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
  # ^^ corrected.
    
  # Determine the (x, y)-coordinates of the intersection rectangle  
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interW = xB - xA + 1
  interH = yB - yA + 1

  # Correction: reject non-overlapping boxes
  if interW <=0 or interH <=0 :
    return -1.0

  interArea = interW * interH
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  iou = interArea / float(boxAArea + boxBArea - interArea)
  return iou


def match_multiple_boxes(boxes_target, boxes_predicted):
    """calculates average IOU score for multiple boxes"""
    total_iou = 0
    for i in range(len(boxes_target)):
        max_iou = 0
        for j in range(len(boxes_predicted)):
            try: 
                curr_iou = bbox_iou(boxes_target[i], boxes_predicted[j])
                if curr_iou > max_iou:
                    max_iou = curr_iou
            except IndexError:
                pass
        total_iou += max_iou
    return total_iou/max(len(boxes_target), len(boxes_predicted))



def collate_fn(batch):
    return tuple(zip(*batch))


def visualize_prediction(imageID, tensor_bounding_box):
    tensor_bounding_box = tensor_bounding_box.cpu().detach().numpy()
    image = cv.imread(f"data/test_images/{imageID}.png")
    image = np.asarray(image)
    for box in tensor_bounding_box:
        x_min = box[0]
        y_min = box[1]
        x_max = box[2]
        y_max = box[3]
        
        color = (0, 0, 255)
        start_point = (int(x_min), int(y_min))
        end_point = (int(x_max), int(y_max))
        thickness = 2
        cv.rectangle(image, start_point, end_point, color, thickness)
    cv.imwrite(f'./results/prediction_{imageID}.png', image)



def visualize_prediction_and_target(imageID, tensor_target, tensor_predicted):
    """Draws labelled and predicted boxes on image"""

    tensor_target = tensor_target.cpu().detach().numpy()
    tensor_predicted = tensor_predicted.cpu().detach().numpy()

    image = cv.imread(f"data/test_images/{imageID}.png")
    image = np.asarray(image)
    for box in tensor_target:
        x_min = box[0]
        y_min = box[1]
        x_max = box[2]
        y_max = box[3]
        
        color = (0, 0, 255) # BLUE: Labeled boxes
        start_point = (int(x_min), int(y_min))
        end_point = (int(x_max), int(y_max))
        thickness = 1
        cv.rectangle(image, start_point, end_point, color, thickness)
    for box in tensor_predicted:
        x_min = box[0]
        y_min = box[1]
        x_max = box[2]
        y_max = box[3]
        
        color = (255, 0, 0) # red: Predicted boxes
        start_point = (int(x_min), int(y_min))
        end_point = (int(x_max), int(y_max))
        thickness = 1
        cv.rectangle(image, start_point, end_point, color, thickness)
    cv.imwrite(f'./results/prediction_{imageID}.png', image)


# Evaluate Prediction
def evaluate_visualize_results(results):
    """Calculates average IOU and visualizes the results"""

    boxes = pd.read_csv("rescaled_data.csv")
    boxes = boxes[["number", "xmin_scaled", "ymin_scaled", "xmax_scaled", "ymax_scaled"]]
    iou_list = []

    for result in results:
        id = result[0]
        # predicted boxes
        predicted_boxes = result[1]
        # target boxes
        sub_df = boxes[boxes["number"] == int(id)]
        num_boxes = len(sub_df)
        box_coordinates = []
        for i in range(num_boxes):
            sub_sub_df = sub_df.iloc[i]
            xmin_scaled = int(sub_sub_df["xmin_scaled"])
            ymin_scaled = int(sub_sub_df["ymin_scaled"])
            xmax_scaled = int(sub_sub_df["xmax_scaled"])
            ymax_scaled = int(sub_sub_df["ymax_scaled"])
            box_coordinates.append(torch.tensor([xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled], dtype=torch.int32))
        if num_boxes > 1:
                target_boxes = torch.stack(box_coordinates, axis=0)
        elif num_boxes == 1:
            box_coordinates = box_coordinates[0]
            target_boxes = box_coordinates.view(1,4)
        else:
            pass

        # evaluate predicted_boxes vs target_boxes
        iou_mean = match_multiple_boxes(target_boxes, predicted_boxes.to("cpu")) 
        iou_list.append(iou_mean)

        # visualize predicted boxes and target boxes
        visualize_prediction_and_target(id, target_boxes, predicted_boxes)

    # calculate iou accross all results
    iou = sum(iou_list) / len(iou_list)
    return iou.item()

    
if __name__=="__main__":

    image_list = os.listdir('./data/test_images/')

    boxes = pd.read_csv("rescaled_data.csv")
    boxes = boxes[["number", "xmin_scaled", "ymin_scaled", "xmax_scaled", "ymax_scaled"]]
    test_data = generate_data(image_list, boxes, "./data/rescaled_png_files/")

    test_dataset = MarginaliaDataset(test_data)
    val_dl = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, pin_memory = True)

    # Load Model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2  # 1 class (marginalia) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model=model.to(device)

    model.load_state_dict(torch.load("faster_r_cnn_weights.pt", map_location=device))


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    results=[]
    detection_threshold = 0.1 # the lower, the less we keep
    model.eval()
    model.to(device)

    with torch.no_grad():
        for images, targets, id in val_dl:    
            try:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(images)

                for i, image in enumerate(images):

                    boxes = outputs[i]['boxes']
                    scores = outputs[i]['scores']
                    labels = outputs[i]['labels']

                    keep = torchvision.ops.nms(boxes, scores, detection_threshold) # the lower, the less we keep
                    boxes = boxes[keep]
                    scores = scores[keep]
                    image_id = id[i]
                
                    op = (id[i], boxes, scores)
                    results.append(op)
            except Exception as e:
                print(e)

    evaluate_visualize_results(results)