import torch
import torchvision
import pandas as pd
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import math
import random


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


# Data Augmentation Functions
def noisy(img, noise_type="gauss"):
    '''
    ### Adding Noise ###
    img: image
    cj_type: {gauss: gaussian, sp: salt & pepper}
    '''
    if noise_type == "gauss":
        image=img.copy() 
        mean=0
        st=0.7
        gauss = np.random.normal(mean,st,image.shape)
        gauss = gauss.astype('uint8')
        image = cv.add(image,gauss)
        return image
    
    elif noise_type == "sp":
        image=img.copy() 
        prob = 0.05
        if len(image.shape) == 2:
            black = 0
            white = 255            
        else:
            colorspace = image.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        probs = np.random.random(image.shape[:2])
        image[probs < (prob / 2)] = black
        image[probs > 1 - (prob / 2)] = white
        return image


def brightness(img, low=0.2, high=0.6):
    """from https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5"""
    value = random.uniform(low, high)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return img

# randomly change the brightness, contrast, and saturation of the image
def colorjitter(img, cj_type="c"):
    '''
    ### Different Color Jitter ###
    img: image
    cj_type: {b: brightness, s: saturation, c: constast}
    '''
    if cj_type == "b":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv.merge((h, s, v))
        img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
        return img
    
    elif cj_type == "s":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv.merge((h, s, v))
        img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
        return img
    
    elif cj_type == "c":
        brightness = 10
        contrast = random.randint(40, 100)
        dummy = np.int16(img)
        dummy = dummy * (contrast/127+1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        img = np.uint8(dummy)
        return img

def data_augmentation(image_list, boxes):
    """Performs data augmentations on train images"""

    augmented_data_df = pd.DataFrame(columns=["number", "xmin_scaled", "ymin_scaled", "xmax_scaled", "ymax_scaled"])
    index = 0
    augmentations = ["flip", "noise", "brightness", "colorjitter"]

    for image in image_list: 
        if image.endswith(".png"): 
            id = image[:-4]
        else: 
            continue
        img = cv.imread(f"./data/rescaled_png_files/{id}.png") # TODO
        
        for augmentation in augmentations:
            
            if augmentation == "flip":
                # horizontal flip
                augmented_img = cv.flip(img, 1)
                new_id = id +"0" # NOTE: Adds 0 to id, to receive a new unique id for augmented img
                cv.imwrite(f'./data/augmented_png_files/{new_id}.png', augmented_img)

                sub_df = boxes[boxes["number"] == int(id)]
                num_boxes = len(sub_df)
                for i in range(num_boxes):
                    sub_sub_df = sub_df.iloc[i]
                    xmin_scaled = int(sub_sub_df["xmin_scaled"])
                    ymin_scaled = int(sub_sub_df["ymin_scaled"])
                    xmax_scaled = int(sub_sub_df["xmax_scaled"])
                    ymax_scaled = int(sub_sub_df["ymax_scaled"])
                    # change coordinates due to flip
                    new_x_min = (500 - xmin_scaled) - abs(xmax_scaled - xmin_scaled)
                    new_x_max = 500 - xmin_scaled

                    # build dict with new data
                    augmented_data_df.loc[index] = [new_id, new_x_min, ymin_scaled, new_x_max, ymax_scaled]
                    index += 1

            if augmentation == "noise":
                augmented_img = noisy(img)
                new_id = id +"1" # NOTE: Adds 0 to id, to receive a new unique id for augmented img
                cv.imwrite(f'./data/augmented_png_files/{new_id}.png', augmented_img)

                sub_df = boxes[boxes["number"] == int(id)]
                num_boxes = len(sub_df)
                for i in range(num_boxes):
                    sub_sub_df = sub_df.iloc[i]
                    xmin_scaled = int(sub_sub_df["xmin_scaled"])
                    ymin_scaled = int(sub_sub_df["ymin_scaled"])
                    xmax_scaled = int(sub_sub_df["xmax_scaled"])
                    ymax_scaled = int(sub_sub_df["ymax_scaled"])
                    # build dict with new data
                    augmented_data_df.loc[index] = [new_id, xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled]
                    index += 1

            if augmentation == "brightness":
                augmented_img = brightness(img)
                new_id = id +"2" # NOTE: Adds 0 to id, to receive a new unique id for augmented img
                cv.imwrite(f'./data/augmented_png_files/{new_id}.png', augmented_img)

                sub_df = boxes[boxes["number"] == int(id)]
                num_boxes = len(sub_df)
                for i in range(num_boxes):
                    sub_sub_df = sub_df.iloc[i]
                    xmin_scaled = int(sub_sub_df["xmin_scaled"])
                    ymin_scaled = int(sub_sub_df["ymin_scaled"])
                    xmax_scaled = int(sub_sub_df["xmax_scaled"])
                    ymax_scaled = int(sub_sub_df["ymax_scaled"])
                    # build dict with new data
                    augmented_data_df.loc[index] = [new_id, xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled]
                    index += 1
            if augmentation == "colorjitter":
                augmented_img = colorjitter(img)
                new_id = id +"3" # NOTE: Adds 0 to id, to receive a new unique id for augmented img
                cv.imwrite(f'./data/augmented_png_files/{new_id}.png', augmented_img)

                sub_df = boxes[boxes["number"] == int(id)]
                num_boxes = len(sub_df)
                for i in range(num_boxes):
                    sub_sub_df = sub_df.iloc[i]
                    xmin_scaled = int(sub_sub_df["xmin_scaled"])
                    ymin_scaled = int(sub_sub_df["ymin_scaled"])
                    xmax_scaled = int(sub_sub_df["xmax_scaled"])
                    ymax_scaled = int(sub_sub_df["ymax_scaled"])
                    # build dict with new data
                    augmented_data_df.loc[index] = [new_id, xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled]
                    index += 1
    return augmented_data_df


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


def collate_fn(batch):
    return tuple(zip(*batch))


# Evaluate Prediction
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


def evaluate_IOU_score(results):
    """Calculates average IOU score"""

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
        # visualize_prediction_and_target(id, target_boxes, predicted_boxes)

    # calculate iou accross all results
    iou = sum(iou_list) / len(iou_list)
    return iou.item()


if __name__=="__main__":
    torch.cuda.empty_cache()

    # original data
    boxes = pd.read_csv("rescaled_data.csv")
    boxes = boxes[["number", "xmin_scaled", "ymin_scaled", "xmax_scaled", "ymax_scaled"]]
    image_list = os.listdir('./data/rescaled_png_files/')

    data_original = generate_data(image_list, boxes, "./data/rescaled_png_files/")

    # train test split
    random.seed(42)
    random.shuffle(data_original)
    num_samples = len(data_original)
    train_size = 0.9

    train_data = data_original[0:math.ceil(train_size*num_samples)]
    test_data = data_original[math.ceil(train_size*num_samples):]
    

    # data augmentation for the training data
    images_to_augment = []
    for data in train_data:
        id = data["image_id"]
        images_to_augment.append(id+".png")

    augmented_data_df = data_augmentation(images_to_augment, boxes)
    augmented_data_df.to_csv("augmented_data.csv")

    boxes_augmented = pd.read_csv("augmented_data.csv")
    boxes_augmented = boxes_augmented[["number", "xmin_scaled", "ymin_scaled", "xmax_scaled", "ymax_scaled"]]

    augmented_image_list = os.listdir("./data/augmented_png_files/")
    data_augmented = generate_data(augmented_image_list, boxes_augmented, image_path="./data/augmented_png_files/")
        

    # combine original and augmented training data
    all_train_data = train_data + data_augmented


    train_dataset = MarginaliaDataset(all_train_data)
    test_dataset = MarginaliaDataset(test_data)

    train_dl = DataLoader(train_dataset, batch_size=4, num_workers=4, collate_fn=collate_fn, pin_memory = True)
    val_dl = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn, pin_memory = True)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2  # 1 class (marginalia) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model=model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    num_epochs = 13

    model.to(device)
    loss_per_epoch = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets, _ in train_dl:
            try:
                optimizer.zero_grad()
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                # print(loss_dict)
                losses = sum(loss for loss in loss_dict.values())
                epoch_loss += losses.item()

                losses.backward()
                optimizer.step()
                torch.cuda.empty_cache()
            except Exception as e:
                print(e)
        try:
            print(f"loss for epoch {epoch}: {epoch_loss / len(train_dl)}")
            loss_per_epoch.append(epoch_loss / len(train_dl))
        except Exception as e:
            print(e)


    # plot IOU score over epochs
    plt.plot(np.arange(len(loss_per_epoch)), np.array(loss_per_epoch))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.savefig("loss_per_epoch.png")
    plt.show()


    # save model
    # torch.save(model.state_dict(), "faster_r_cnn_weights.pt")

