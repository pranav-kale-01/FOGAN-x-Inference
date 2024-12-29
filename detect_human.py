import cv2
import imutils
import numpy as np 

protopath = "./hd_model/MobileNetSSD_deploy.prototxt"
modelpath = "./hd_model/MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protopath, modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# cap = cv2.VideoCapture("videos/test_video.mp4")
def inferVideo():
    cap = cv2.VideoCapture("videos/test_video.mp4")
    # cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        # print(type(frame))
        frame = imutils.resize(frame, width=800)

        (H, W) = frame.shape[:2]
        # print(frame.shape)
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        # print(blob.shape)
        detector.setInput(blob)
        person_detections = detector.forward()

        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])
                if CLASSES[idx] != "person":
                    continue
                
                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        cv2.imshow("Application", frame)
        key = cv2.waitKey(100)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


import os 
def inferImage():
        img = np.random.choice(os.listdir("./videos/"))
        print(img)
        frame=cv2.imread("./videos/"+img)
        # frame=cv2.imread("./videos/2.png")
        frame = imutils.resize(frame, width=800)

        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        
        detector.setInput(blob)
        person_detections = detector.forward()

        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])
                if CLASSES[idx] != "person":
                    continue
                
                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        cv2.imshow("Application", frame)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()

inferVideo()