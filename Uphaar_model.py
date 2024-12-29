import torch 
import cv2
from PIL import Image
import torchvision.transforms as transforms 
import time
from torchvision.utils import save_image
from generator import GeneratorResNet
import os
import numpy as np
import scipy.misc
import imutils
from PIL import Image
from PIL import Image as im 
import threading

def save_img( frame ):
    # frame = im.fromarray(frame)
    # frame.show()
    # frame= output[0].numpy()[:, ::-1, :, :]
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
    key = cv2.waitKey(int(100/15))        
    if key == ord("q"):
        return




protopath = "./hd_model/MobileNetSSD_deploy.prototxt"
modelpath = "./hd_model/MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protopath, modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


loaded_weights = torch.load('./saved_models/generator.pt') #, map_location=torch.device('cpu') )

model = GeneratorResNet(input_nc=3, output_nc=3, ngf=64, n_blocks=4,img_size=512, light=True)
model.load_state_dict( loaded_weights )
model.eval()
model.to("cuda")
# cam = cv2.VideoCapture("https://172.16.141.122:8080/video")
cam = cv2.VideoCapture("https://192.168.89.36:8080/video")

# cam = cv2.VideoCapture("videos/prev.mp4")

transform = transforms.Compose([ 
            transforms.Resize((512, 512)),
            transforms.ToTensor() 
        ]) 

count = 0 
while True:
    ret, frame = cam.read()
    
    with torch.no_grad():
        if( count > 3 and count % 3 == 0 ):
            img_tensor = transform(Image.fromarray(np.array(frame)[:,:,::-1])).unsqueeze(0).cuda()
            output  = model(img_tensor)
            output = output[0]
            save_image(output[0], f"val_outputs/out.png", normalize=True)
            frame = cv2.imread("val_outputs/out.png")

            t1 = threading.Thread( target= save_img(frame) )
            t1.start()
        count += 1

cv2.destroyAllWindows()