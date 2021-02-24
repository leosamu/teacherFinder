import cv2
import pandas as pd
import numpy as np
import pdb

COCONAMESPATH='darknet/data/coco.names'
YOLOCFGPATH='yolov3.cfg'
YOLOWEIGHTPATH='yolov3.weights'
THRESHOLD=1

def extractImages(pathIn,numFrames):
    """extract a set of equidistant images in a video located in pathIn

    Args:
        pathIn (string): path of the processed video
        numFrames (int): number of frames requested

    Returns:
        [image]: an array of images that will be processed later with darknet
    """
    count=0
    imageList=[]
    vidcap = cv2.VideoCapture(pathIn)    
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES,(count*vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/numFrames))    # get a frame every 10% of the video
        success,image = vidcap.read()
        imageList.append(image)
        count=count+1
    return imageList
        
def hasPerson(imageList):
    """check a list of images if it has persons in it or not

    Args:
        imageList ([images]): array of images that needs to be treated

    Returns:
        [int]: array with 0 or 1 as the result of the processing of the 
        image array
    """
    #initialize the person array to store 1 or 0 depending if we found a person
    people=[]
    #initialize the classes array
    classes = None    
    with open(COCONAMESPATH, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    #initialize the neural network    
    for image in imageList:        
        if (image is None)==False:
            #maybe we can move this out to make it faster
            net = cv2.dnn.readNet(YOLOWEIGHTPATH, YOLOCFGPATH)
            net.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            outs = net.forward(output_layers)
            class_ids = []
            confidences = []
            boxes = []
            Width = image.shape[1]
            Height = image.shape[0]
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.1:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])
            #0 is the classid of person in the coco.names files
            if 0 in class_ids:
                people.append(1)
            else:
                people.append(0)    
    return people

def return_code(file,frames):
    """Open a video and pick a number of equidistant frames wich will be treated with darkent to check if there 
        are persons in it will return True if the number of persons founds are higher than the TRESHOLD defined at
        the beggining of the file.

    Args:
        file (string): path to the processed video
        frames (int): number of frames that will be checked in the video 

    Returns:
        dict: a dictionary with the format {'estado':True,'error':0} if error > 0 means that something went wrong, 
        estado will show if enought persons has been found in the video
    """    
    found = False
    error = 0
    try:
        imageList = extractImages(file,frames)
        people = hasPerson(imageList)        
        countPeople = sum(people)
        if countPeople > THRESHOLD:
            found=True
    except:
        error=1
    return {'estado':found,'error':error}

if __name__ == "__main__":
    print(return_code('/Volumes/vosupv/profechecker/fdb0af19-e074-4491-8cb8-0ffdfe7562de.mp4',10))