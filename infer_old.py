import torch 
import cv2
from PIL import Image
import torchvision.transforms as transforms 
import time
from torchvision.utils import save_image
from generator import GeneratorResNet
import os
import numpy as np
import imutils
from PIL import Image
import threading
import queue

protopath = "./hd_model/MobileNetSSD_deploy.prototxt"
modelpath = "./hd_model/MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protopath, modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

image_queue = queue.Queue()
stop_event = threading.Event()

def save_images_worker():
    """Worker function that saves images from the queue in a separate thread."""
    while not stop_event.is_set() or not image_queue.empty():
        try:
            print("saving image")
            tensor, filename = image_queue.get(timeout=1)  # Waits for a new item
            save_image(tensor.clone().detach(), filename)  # Save the image
            image_queue.task_done()
        except queue.Empty:
            pass  # If empty, check again

# Start image saving thread
save_thread = threading.Thread(target=save_images_worker, daemon=True)
save_thread.start()

def infer_export_video(video_name):
    try:
        # Check CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Check PyTorch installation.")
        device = torch.device("cuda")

        # Load model
        model = GeneratorResNet(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=512, light=True)
        model.load_state_dict(torch.load('./saved_models/generator.pt', map_location=device))
        model.eval().to(device)
        print(f"Model device: {next(model.parameters()).device}")  # Verify

        # Load video
        cam = cv2.VideoCapture(video_name)
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_video.mp4', fourcc, 30, (800, 800))

        currentframe = 0
        while cam.isOpened():
            start_time = time.time()

            ret, frame = cam.read()
            if not ret:
                break

    
            # Convert frame to tensor and move to GPU
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = transform(frame_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)

                # Extract the correct tensor from the tuple
                if isinstance(output, tuple):
                    output_tensor = output[0]  
                else:
                    output_tensor = output 
            

            # Queue the image saving task in a separate thread
            image_queue.put((img_tensor.cpu(), f"./val_outputs/in.png"))
            image_queue.put((output_tensor.cpu(), f"./val_outputs/out.png"))

            # Convert tensor to numpy array
            output_img = output_tensor[0].cpu().detach().numpy()

            # Ensure the output is in (H, W, 3) format
            if output_img.shape[0] == 3:  # Shape is (3, H, W)
                output_img = np.transpose(output_img, (1, 2, 0))
            else:
                raise ValueError(f"Unexpected output shape: {output_img.shape}")

            # Scale to 0-255 and convert to uint8
            output_img = (output_img * 255).astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            output_img = imutils.resize(output_img, width=800)

            # Display in real-time
            cv2.imshow("Desmoked Video", output_img)

            print(f"inference time - {time.time() - start_time }")
            
            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


            # Write to video
            #out.write(output_img)
            print(f"Processed frame: {currentframe}")
            
            currentframe += 1

    except OSError as oe:
        print(f"Error: Creating directory of data {oe}")
    except Exception as e:
        print(f"Exception occurred while inferring video {e}")
        raise e

    cam.release()
    out.release() # releasing the VideoWriter object
    cv2.destroyAllWindows()

    # Stop the image-saving thread after inference is done
    stop_event.set()
    save_thread.join()

infer_export_video("./videos/smoky_video.mp4")
# export_onnx()
# optimize_onnx()   
# infer_onnx_video("./videos/smoky_video.mp4")
# check_onnx_supports_cuda()