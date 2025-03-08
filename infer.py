import threading
import queue
import time
import cv2
import torch
import numpy as np
from collections import deque
from generator import GeneratorResNet
from concurrent.futures import ThreadPoolExecutor

# -------------------- QUEUES & EVENTS --------------------
frame_queue = queue.Queue(maxsize=10)   # Stores frames for processing
result_queue = queue.Queue(maxsize=20)       # Stores processed frames for display
stop_event = threading.Event()          # Used to signal threads to stop

TARGET_FPS=30
FRAME_INTERVAL = 1.0 / TARGET_FPS  # 1/30 = 0.033s per frame

# -------------------- MODEL LOADING --------------------
def load_model():
    print("DEBUG: [Model] Loading model...")
    model = GeneratorResNet(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=512, light=True)
    model.load_state_dict(torch.load('./saved_models/generator.pt', map_location="cuda"))
    model.eval().to("cuda")
    print("DEBUG: [Model] Model loaded successfully!")
    return model

# -------------------- FRAME PREPROCESSING --------------------
def preprocess_frame(frame):
    """ Convert OpenCV frame to PyTorch tensor. """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame_resized = cv2.resize(frame_rgb, (512, 512))   # Resize to 512x512
    img_tensor = torch.tensor(frame_resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Normalize & reshape
    img_tensor = img_tensor.to("cuda")
    print("DEBUG: [Preprocess] Frame converted to tensor.")
    return img_tensor

# -------------------- POSTPROCESSING --------------------
def postprocess_frame(input_tensor, output_tensor):
    """ Convert model output tensor to OpenCV image. """
    input_img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)  # Convert to (H, W, 3)
    output_img = output_tensor[0].cpu().detach().numpy().transpose(1, 2, 0)  # Convert model output

    if input_img.shape[0] == 3:  # Shape is (3, H, W)
        input_tensor = np.transpose(input_img, (1, 2, 0))
    if output_img.shape[0] == 3:  # Shape is (3, H, W)
        output_img = np.transpose(output_img, (1, 2, 0))

    # Convert to OpenCV format
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    combined = np.hstack((input_img, output_img))  # Side-by-side view
    print("DEBUG: [Postprocess] Frames combined successfully.")
    return combined

# -------------------- INFERENCE WORKER --------------------
def infer_worker(model):
    """ Worker thread that processes frames from frame_queue. """
    print("DEBUG: [Worker] Inference worker started.")
    
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.05)
            print("DEBUG: [Worker] Frame pulled from queue.")
        except queue.Empty:
            continue  

        start_time = time.time()  # Track inference start time
        img_tensor = preprocess_frame(frame)

        with torch.no_grad():
            output_tensor = model(img_tensor)
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]  

        result_queue.put(postprocess_frame(img_tensor.cpu(), output_tensor.cpu()))  
        print(f"DEBUG: [Worker] Inference complete in {time.time() - start_time:.4f}s")

        frame_queue.task_done()  

    print("DEBUG: [Worker] Inference worker stopped.")

# -------------------- DISPLAY WORKER --------------------
def show_side_by_side():
    """ Displays processed frames side by side. """
    print("DEBUG: [Display] Display worker started.")
    last_display_time = time.time()

    while not stop_event.is_set():
        try:
            img = result_queue.get(timeout=1)
            print("DEBUG: [Display] Frame pulled from result queue.")
        except queue.Empty:
            continue  

        current_time = time.time()
        time_since_last_frame = current_time - last_display_time

        # Ensure ~30 FPS by waiting if needed
        if time_since_last_frame < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - time_since_last_frame)

        cv2.imshow("Processed Video", img)
        last_display_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("DEBUG: [Display] 'q' pressed. Stopping display.")
            stop_event.set()

    cv2.destroyAllWindows()
    print("DEBUG: [Display] Display worker stopped.")

# -------------------- MAIN FUNCTION --------------------
def main():
    print("DEBUG: [Main] Starting video processing...")
    model = load_model()  

    cam = cv2.VideoCapture("./videos/smoky_video.mp4")

    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(infer_worker, model)  
        executor.submit(infer_worker, model)  
        executor.submit(show_side_by_side)    

        try:
            while cam.isOpened():
                ret, frame = cam.read()
                if not ret:
                    print("DEBUG: [Main] End of video reached.")
                    break

                # Ensure we process every frame
                while frame_queue.full():
                    print("DEBUG: [Main] Queue full. Waiting for space...")

                frame_queue.put(frame)
                print(f"DEBUG: [Main] Frame added to queue (Queue Size: {frame_queue.qsize()})")

        except KeyboardInterrupt:
            print("DEBUG: [Main] KeyboardInterrupt detected, stopping...")
            stop_event.set()  

        finally:
            cam.release()
            print("DEBUG: [Main] Video processing stopped.")

            # ---- FIX: Wait for workers to finish before exiting ----
            frame_queue.join()  
            result_queue.join()  
            stop_event.set()  
            time.sleep(1)  # Let threads finish

if __name__ == "__main__":
    main()
