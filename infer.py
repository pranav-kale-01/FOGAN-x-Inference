import threading
import queue
import time
import cv2
import torch
import numpy as np
from generator import GeneratorResNet
from concurrent.futures import ThreadPoolExecutor

# -------------------- QUEUES & EVENTS --------------------
frame_queue = queue.Queue(maxsize=5)  # Stores tuples: (frame_number, frame)
# Use PriorityQueue for ordered results based on frame number.
result_queue = queue.PriorityQueue(maxsize=5)  # Stores tuples: (frame_number, processed_frame)
stop_event = threading.Event()  # Used to signal threads to stop

TARGET_FPS = 8
FRAME_INTERVAL = 1.0 / TARGET_FPS  

# -------------------- MODEL LOADING --------------------
def load_model():
    print("DEBUG: [Model] Loading model...")
    model = GeneratorResNet(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=512, light=True)
    model.load_state_dict(torch.load('./saved_models/generator.pt', map_location="cuda"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model to GPU
    model.to(device).eval()
    print("DEBUG: [Model] Model loaded successfully!")
    return model

# -------------------- FRAME PREPROCESSING --------------------
def preprocess_frame(frame):
    """ Convert OpenCV frame to PyTorch tensor. """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame_resized = cv2.resize(frame_rgb, (512, 512))     # Resize to 512x512
    img_tensor = torch.tensor(frame_resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Normalize & reshape
    img_tensor = img_tensor.to("cuda")
    print("DEBUG: [Preprocess] Frame converted to tensor.")
    return img_tensor

# -------------------- POSTPROCESSING --------------------
def postprocess_frame(input_tensor, output_tensor):
    """ Convert model output tensor to OpenCV image. """
    input_img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)   # (H, W, 3)
    output_img = output_tensor[0].cpu().detach().numpy().transpose(1, 2, 0)
    
    # If transposing is needed (sometimes redundant)
    if input_img.shape[0] == 3:
        input_img = np.transpose(input_img, (1, 2, 0))
    if output_img.shape[0] == 3:
        output_img = np.transpose(output_img, (1, 2, 0))
    
    # Convert to OpenCV format
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    
    combined = np.hstack((input_img, output_img))  # Side-by-side view
    print("DEBUG: [Postprocess] Frames combined successfully.")
    return combined

# -------------------- INFERENCE WORKER --------------------
def infer_worker(model, worker_id):
    """ Worker thread that processes frames from frame_queue. """
    print(f"DEBUG: [Worker {worker_id}] Inference worker started.")
    while not stop_event.is_set():
        try:
            item = frame_queue.get()  # item is a tuple: (frame_number, frame)
            print(f"DEBUG: [Worker {worker_id}] Frame pulled from queue.")
            if item is None or item[0] is None:
                print(f"DEBUG: [Worker {worker_id}] Sentinel received, stopping.")
                frame_queue.task_done()
                break
            frame_number, frame = item
        except queue.Empty:
            continue

        start_time = time.time()  # Track inference start time
        img_tensor = preprocess_frame(frame)

        with torch.no_grad():
            output_tensor = model(img_tensor)
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]
            torch.cuda.synchronize()

            # Put the result into the result_queue along with the frame_number
            processed_frame = postprocess_frame(img_tensor.cuda(), output_tensor.cuda())
            result_queue.put((frame_number, processed_frame))
            print(f"DEBUG: [Worker {worker_id}] Inference complete in {time.time() - start_time:.4f}s")

        frame_queue.task_done()

    print(f"DEBUG: [Worker {worker_id}] Inference worker stopped.")

# -------------------- DISPLAY WORKER --------------------
def show_side_by_side():
    """ Displays processed frames side by side in order. """
    print("DEBUG: [Display] Display worker started.")
    last_display_time = time.time()
    expected_frame = 0  # The next frame number expected for display

    # Buffer for out-of-order frames
    buffer = {}

    while not stop_event.is_set():
        try:
              # Get the next available processed frame (ordered by frame number)
            frame_number, img = result_queue.get()
            print(f"DEBUG: [Display] Received frame {frame_number} from result queue.")
            result_queue.task_done()

            if frame_number is None:
                print("DEBUG: [Display] Sentinel received, stopping.")
                break

            # Resize the combined image so that each half is 800x800
            # (Assuming the combined image is a horizontal concatenation of two images)
            img_resized = cv2.resize(img, (1600, 800))

            # If the frame is not the expected one, store it in the buffer.
            if frame_number != expected_frame:
                buffer[frame_number] = img_resized
                # Try to output any buffered frames that are now in order.
                while expected_frame in buffer:
                    cv2.imshow("Processed Video", buffer.pop(expected_frame))
                    expected_frame += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("DEBUG: [Display] 'q' pressed. Stopping display.")
                        stop_event.set()
                        break
            else:
                # If the frame is the expected one, display it
                cv2.imshow("Processed Video", img_resized)
                expected_frame += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("DEBUG: [Display] 'q' pressed. Stopping display.")
                    stop_event.set()
                    break

            # Maintain target FPS if needed
            current_time = time.time()
            time_since_last_frame = current_time - last_display_time
            if time_since_last_frame < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - time_since_last_frame)
            last_display_time = time.time()

        except queue.Empty:
            continue
        except Exception as e:
            raise e

    cv2.destroyAllWindows()
    print("DEBUG: [Display] Display worker stopped.")

# -------------------- MAIN FUNCTION --------------------
def main():
    print("DEBUG: [Main] Starting video processing...")
    model = load_model()
    print(next(model.parameters()).device)

    cam = cv2.VideoCapture("./videos/smoky_video.mp4")
    frame_id = 0

    # Record start time for FPS calculation
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Start inference workers with unique IDs
        executor.submit(infer_worker, model, 1)
        executor.submit(infer_worker, model, 2)
        executor.submit(infer_worker, model, 3)
        executor.submit(show_side_by_side)

        try:
            frame_counter = 0
            while cam.isOpened() and not stop_event.is_set():
                ret, frame = cam.read()
                if not ret:
                    print("DEBUG: [Main] End of video reached.")
                    break

                if frame_counter % 3 == 0:
                    frame_counter += 1
                    continue

                # Wait if the frame_queue is full (with a timeout)
                wait_start = time.time()
                while frame_queue.full() and not stop_event.is_set():
                    if time.time() - wait_start > 1:
                        break
                    print("DEBUG: [Main] Queue full. Waiting for space...")
                    time.sleep(0.1)
                if stop_event.is_set():
                    break

                # Enqueue the frame with its frame number
                try:
                    frame_queue.put((frame_id, frame), timeout=0.02)
                    print(f"DEBUG: [Main] Frame {frame_id} added to queue (Queue Size: {frame_queue.qsize()})")
                    frame_id += 1
                except queue.Full:
                    print("DEBUG: [Main] Failed to add frame - queue still full, skipping frame.")
                    break

                frame_counter += 1

            print("DEBUG: [Main] Exiting main loop.")
            
        except KeyboardInterrupt:
            print("DEBUG: [Main] KeyboardInterrupt detected, stopping...")
            stop_event.set()

        finally:
            stop_event.set()
            cam.release()
            print("DEBUG: [Main] Video processing stopped.")

            # Send sentinels to stop the workers
            for _ in range(3):
                try:
                    frame_queue.put_nowait((None, None))
                except queue.Full:
                    pass
            try:
                result_queue.put_nowait((None, None))
            except queue.Full:
                pass

            executor.shutdown(wait=False)
            print("DEBUG: [Main] Executor shutdown initiated.")

            # Calculate and print the FPS based on the number of frames enqueued and elapsed time.
            elapsed_time = time.time() - start_time
            fps = frame_id / elapsed_time if elapsed_time > 0 else 0
            
            print(f"DEBUG: [Main] Total frames enqueued: {frame_id}")
            print(f"DEBUG: [Main] Elapsed time: {elapsed_time:.2f} seconds")
            print(f"DEBUG: [Main] Approximate processing FPS: {fps:.2f}")

if __name__ == "__main__":
    main()
