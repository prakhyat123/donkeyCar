import argparse
import os
from pathlib import Path
import cv2
import numpy as np

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

#image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

#https://www.hackster.io/kemfic/simple-lane-detection-c3db2f
def color_filter(image):
    #convert to HLS to mask based on HLS
    image = np.array(image)
    image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower = np.array([0,190,0])
    upper = np.array([255,255,255])
    yellower = np.array([10,0,90])
    yelupper = np.array([50,255,255])
    yellowmask = cv2.inRange(hls, yellower, yelupper)
    whitemask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellowmask, whitemask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def color_filter_ori(image):
    #convert to HLS to mask based on HLS
    image = np.array(image)
    #image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    hlsImage = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hChannel, lChannel, sChannel = cv2.split(hlsImage)
    lowerThreshold = 65
    higherThreshold = 255
    returnValue, binaryThresholdedImage = cv2.threshold(sChannel,lowerThreshold,higherThreshold,cv2.THRESH_BINARY)
    thresholdedImage = cv2.cvtColor(binaryThresholdedImage, cv2.COLOR_GRAY2RGB)
    return thresholdedImage

def color_filter_ori_color(image):
    #convert to HLS to mask based on HLS
    image = np.array(image)
    #image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    hlsImage = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hChannel, lChannel, sChannel = cv2.split(hlsImage)
    lowerThreshold = 65
    higherThreshold = 255
    returnValue, binaryThresholdedImage = cv2.threshold(sChannel,lowerThreshold,higherThreshold,cv2.THRESH_BINARY)
    masked = cv2.bitwise_and(image, image, mask = binaryThresholdedImage)
    return masked

def color_filter_ori_random_color(image):
    #convert to HLS to mask based on HLS
    image = np.array(image)
    random_color = list(np.random.choice(range(256), size=3))
    single_color = np.zeros(image.shape, np.uint8)
    single_color[:] = random_color
    #image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    hlsImage = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hChannel, lChannel, sChannel = cv2.split(hlsImage)
    lowerThreshold = 65
    higherThreshold = 255
    returnValue, binaryThresholdedImage = cv2.threshold(sChannel,lowerThreshold,higherThreshold,cv2.THRESH_BINARY)
    thresholdedImage = cv2.cvtColor(binaryThresholdedImage, cv2.COLOR_GRAY2RGB)
    masked = cv2.bitwise_and(single_color, single_color, mask = binaryThresholdedImage)
    return masked

def main(args):
    print("Hello World")
    inPath = args.input_dir
    outPath = args.output_dir

    if not os.path.exists(outPath):
        os.makedirs(outPath)

    for imagePath in os.listdir(inPath):
        #image input path
        imgPath = os.path.join(inPath, imagePath)
        #Convert the image
        img = cv2.imread(imgPath)
        img = color_filter_ori_random_color(img)
        #image output path
        imgOutPath = os.path.join(outPath, 'filter_'+imagePath)
        #Save image in the output path
        print(imgOutPath)
        cv2.imwrite(imgOutPath, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Representation Learning for DonkeyCar")
    parser.add_argument("--input-dir", type=Path, required=True, help='Path to the input Images')
    parser.add_argument("--output-dir", type=Path, required=True, help='Path to the output Image folder ')
    args = parser.parse_args()
    main(args)





