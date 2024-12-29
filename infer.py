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

protopath = "./hd_model/MobileNetSSD_deploy.prototxt"
modelpath = "./hd_model/MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protopath, modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


def infer_export_video( video_name ): 
    is_GPU = torch.cuda.is_available() 
        
    if( not is_GPU ):
        loaded_weights = torch.load('./saved_models/generator.pt' , map_location=torch.device('cpu') )
    else: 
        loaded_weights = torch.load('./saved_models/generator.pt')

    model = GeneratorResNet(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=512, light=True)
    model.load_state_dict( loaded_weights )
    model.eval() 
    model.to("cuda") if is_GPU else model.to("cpu")
    

    # loading the video 
    cam = cv2.VideoCapture(video_name)
  
    transform = transforms.Compose([ 
        transforms.Resize((512, 512)),
        transforms.ToTensor() 
    ]) 

    try: 
        if not os.path.exists("data"):
            os.makedirs('data')
            
        # Get the video's width, height, and FPS for the output video
        fps = 30
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (800, 800))

        # frames 
        currentframe = 0 
        while( True ): 
            ret, frame = cam.read()

            if ret: 
                if( currentframe <= 300):
                    start_time = time.time()

                    with torch.no_grad():    
                        img_tensor = transform( Image.fromarray(np.array(frame)[:,:,::-1]) ).unsqueeze(0)
                        
                        # loading to either gpu or cpu based on what is selected
                        img_tensor = img_tensor.cuda() if is_GPU else img_tensor.cpu()
                        output = model( img_tensor ) 
                        
                    
                        save_image( img_tensor, f"val_outputs/in.png", normalize=True)
                        save_image( output[0], f"val_outputs/out.png", normalize=True)
                        end_time = time.time()

                        print("Current Frame - ", currentframe )
                        print( "Inference time - ", (end_time - start_time ) ) 

                        frame =  cv2.imread("val_outputs/out.png")
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
                        cv2.imwrite("bounded.jpg",frame)
                        
                        # Ensure dimensions are consistent
                        assert frame.shape[1] == 800 and frame.shape[0] == 800, "Frame dimensions are incorrect!"

                        # Write the processed frame to the video
                        out.write(frame)
                

                currentframe += 1
            else: 
                break

    except OSError: 
        print("Error: Creaint directory of data")

    cam.release()
    out.release() # releasing the VideoWriter object
    cv2.destroyAllWindows()

# video_name = '/content/drive/MyDrive/video.mp4'
def infer_video( video_name ):
    is_GPU = torch.cuda.is_available() 
        
    if( not is_GPU ):
        loaded_weights = torch.load('./saved_models/generator.pt' , map_location=torch.device('cpu') )
    else: 
        loaded_weights = torch.load('./saved_models/generator.pt')

    model = GeneratorResNet(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=512, light=True)
    model.load_state_dict( loaded_weights )
    model.eval() 
    model.to("cuda") if is_GPU else model.to("cpu")
    

    # loading the video 
    cam = cv2.VideoCapture(video_name)
    # cam = cv2.VideoCapture(0)
    transform = transforms.Compose([ 
        transforms.Resize((512, 512)),
        transforms.ToTensor() 
    ]) 

    try: 
        if not os.path.exists("data"):
            os.makedirs('data')

        # frames 
        currentframe = 0 
        while( True ): 
            ret, frame = cam.read()

            if ret: 
                if( currentframe > 4 and currentframe % 4 == 0 ):
                    start_time = time.time()

                    with torch.no_grad():    
                        img_tensor = transform( Image.fromarray(np.array(frame)[:,:,::-1]) ).unsqueeze(0)
                        
                        # loading to either gpu or cpu based on what is selected
                        img_tensor = img_tensor.cuda() if is_GPU else img_tensor.cpu()
                        output = model( img_tensor ) 

                        
                        save_image( img_tensor, f"val_outputs/in.png", normalize=True)
                        save_image( output[0], f"val_outputs/out.png", normalize=True)
                        end_time = time.time()
                        print( "Inference time - ", (end_time - start_time ) ) 

                        frame =  cv2.imread("val_outputs/out.png")
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
                        cv2.imwrite("bounded.jpg",frame)
                        # frame = cv2.imread("val_outputs/out.png")
                        # # frame = frame.transpose((2, 3, 1, 0)).reshape((512, 512, 3))

                        # frame = imutils.resize(frame,width=512)
                       
                        # (H,W) = frame.shape[:2]
                       
                        # blob = cv2.dnn.blobFromImage(frame,0.007843,(W,H),127.5)
                        # # cv2.imshow()
                        # detector.setInput(blob)
                        # person_detections = detector.forward()
                        # for i in np.arange(0, person_detections.shape[2]):
                        #     confidence = person_detections[0, 0, i, 2]
                        #     if confidence > 0.5:
                        #         idx = int(person_detections[0, 0, i, 1])
                        #         if CLASSES[idx] != "person":
                        #             continue
                                
                        #         person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        #         (startX, startY, endX, endY) = person_box.astype("int")
                        #         cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                        #         print("Person Detected")

                        #     cv2.imshow("App",frame)
                        #     cv2.waitKey(0)


                        #     cv2.destroyAllWindows()

                currentframe += 1
            else: 
                break

    except OSError: 
        print("Error: Creaint directory of data")

    cam.release()
    cv2.destroyAllWindows()

def infer_image( img_name ):    
    try: 
        loaded_weights = torch.load('./saved_models/generator.pt') #, map_location=torch.device('cpu') )
        transform = transforms.Compose([ 
            transforms.Resize((512, 512)),
            transforms.ToTensor() 
        ])     

        model = GeneratorResNet(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=512, light=True)
        model.load_state_dict( loaded_weights )
        model.to("cuda")
        model.eval()

        start_time = time.time()
        img_tensor =  transform( Image.fromarray(np.array(Image.open(f"./input/{img_name}.png"))[:,:,::-1]) ).unsqueeze(0).cuda()
        output = model( img_tensor ) 
        save_image( img_tensor, f"val_outputs/{img_name}_in.png", normalize=True)
        save_image( output[0], f"val_outputs/{img_name}_out.png", normalize=True)
        end_time = time.time()

        print( "Inference time - ", (end_time - start_time ) ) 

    except Exception as e: 
        print("error", e )
        
infer_export_video("./videos/prev.mp4")
    