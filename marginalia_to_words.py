from skimage.io import imread
from skimage.color import rgb2gray
#import matplotlib.pyplot as plt
import cv2
from skimage.filters import sobel
import numpy as np
import os
from skimage.filters import threshold_otsu

current_folder_path = "/cephyr/users/adamax/Alvis/Project-Marginalia/model-to-words/"
marginalia_folder_path = "/cephyr/users/adamax/Alvis/Project-Marginalia/model-to-words/marginalia/"
output_folder_path = "/cephyr/users/adamax/Alvis/Project-Marginalia/model-to-words/output_words/"

marginalia_list = os.listdir(marginalia_folder_path)
print(marginalia_list)

def horizontal_projections(sobel_image):
    return np.sum(sobel_image, axis=1)  

def vertical_projections(sobel_image):
    return np.sum(sobel_image, axis=0)  

def find_peak_regions(hpp, divider=2):
    threshold = (np.max(hpp)-np.min(hpp))/divider
    peaks = []
    peaks_index = []
    for i, hppv in enumerate(hpp):
        if hppv < threshold:
            peaks.append([i, hppv])
    return peaks

def thresholding(image):
    #img_grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY)
    #plt.imshow(thresh, cmap='gray')
    return thresh

for marginalia in marginalia_list:
    img = (imread(marginalia_folder_path + marginalia))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height, width = img.shape
    sobel_image = sobel(img)
    #Horizontal projection to see where the lines are
    hpp = horizontal_projections(sobel_image)
    #See where there is white-space
    peaks = find_peak_regions(hpp)
    peaks_index = np.array(peaks).astype(int)
    img_copy = np.copy(img)

    #Fill white space to black
    r,c = img_copy.shape
    for ri in range(r):
        if ri in peaks_index:
            img_copy[ri, :] = 0

    #black-white instead of gray-scale
    thresh_img = thresholding(img_copy)
    #find the contours of our lines
    (contours, heirarchy) = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) # (x, y, w, h)

    lines = []
    img2 = img.copy()
    for i,ctr in enumerate(sorted_contours_lines):
        x,y,w,h = cv2.boundingRect(ctr)
        #filter out lines that are too small
        if ((w*h)/(width*height)) > 0.05:
            #save the lines of the marginalia
            lines.append(img2[y:y+h, x:x+w])
    
    #find each word in each line
    word_counter = 0
    for line in lines:
        #black/white
        thresh = threshold_otsu(line)
        binary = line > thresh 
        vertical_projection = vertical_projections(binary)

        height = line.shape[0]
        ## we will go through the vertical projections and 
        ## find the sequence of consecutive white spaces in the image
        whitespace_lengths = []
        whitespace = 0
        for vp in vertical_projection:
            if vp == height:
                whitespace = whitespace + 1
            elif vp != height:
                if whitespace != 0:
                    whitespace_lengths.append(whitespace)
                    whitespace = 0 # reset whitepsace counter.

        avg_white_space_length = np.mean(whitespace_lengths)

        whitespace_length = 0
        divider_indexes = []
        for index, vp in enumerate(vertical_projection):
            if vp >= height:
                whitespace_length = whitespace_length + 1
            elif vp != height:
                if whitespace_length != 0 and whitespace_length > avg_white_space_length:
                    divider_indexes.append(index-int(whitespace_length/2))
                    whitespace_length = 0 # reset it

        divider_indexes = np.array(divider_indexes)
        current_index = 0
        if len(divider_indexes) == 0:
            cv2.imwrite(output_folder_path + marginalia[:-4] + "_word_" + str(word_counter) + ".png", line)
            word_counter += 1
        else:
            for i, index in enumerate(divider_indexes):
                word = line[0:,current_index:index]
                current_index = index
                cv2.imwrite(output_folder_path + marginalia[:-4] + "_word_" + str(word_counter) + ".png", word)
                word_counter += 1
                if i==len(divider_indexes)-1:
                    word = line[0:,current_index:]
                    cv2.imwrite(output_folder_path + marginalia[:-4] + "_word_" + str(word_counter) + ".png", word)
