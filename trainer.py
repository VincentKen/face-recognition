import cv2
import numpy as np
import os
import json

def normalizer(src):

    if len(src.shape) == 2: # if the shape is two it has a single channel
        dst = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    elif len(src.shape) == 3 and src.shape[2] == 3: # if the length is 3 then the amount of channels can be found in the third element
        dst = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
    else:
        dst = np.copy(src)
    return dst

def read_dirs():
    images = []
    labels = {
        "labels_int": [],
        "labels_str": []
    }

    curr_label = 0
    prefix = "images/"
    directories = os.listdir(prefix)
    for dir in directories:
        if not os.path.isdir(prefix + dir):
            pass
        for f in os.listdir(prefix+dir):
            file = os.path.join(prefix+dir, f)
            if not os.path.isfile(file):
                pass
            images.append(cv2.resize(cv2.imread(file, 0), (250, 250)))
            labels["labels_int"].append(curr_label)
            labels["labels_str"].append(dir)
        curr_label+=1
    return images, labels


def save_labels(labels):
    new_labels = {}
    prev_lab = -1
    for i in range(0, len(labels["labels_int"])):
        if labels["labels_int"][i] == prev_lab:
            continue
        new_labels[labels["labels_int"][i]] = labels["labels_str"][i]
        prev_lab = labels["labels_int"][i]
    with open("labels.json", 'w') as file:
        json.dump(new_labels, file)


if __name__ == "__main__":
    images, labels = read_dirs()
    height = np.size(images[0], 0)
    
    testSample = images.pop()
    testLabel = labels["labels_int"].pop()
    testLabelStr = labels["labels_str"].pop()
    
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.train(images, np.array(labels["labels_int"]))

    predictedLabel = recognizer.predict(testSample)
    
    if predictedLabel[0] is not testLabel:
        print("Predicted label and actual label do not match up")
    else:
        save_loc = "recognizer.xml"
        print("Training succesful, saving to %s" % save_loc)
        recognizer.save(save_loc)
        save_labels(labels)
    # print("Got %d, should have been %d" % (predictedLabel, testLabel))
    
