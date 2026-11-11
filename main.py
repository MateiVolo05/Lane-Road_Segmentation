import os
import queue
import shutil
from threading import Thread
import torch
from lib.models.laneatt import LaneATT
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = LaneATT(
    backbone='resnet34',
    topk_anchors=1000,
    anchors_freq_path='LaneATT/data/tusimple_anchors_freq.pt'
)
model_path = "LaneATT/experiments/laneatt_r34_tusimple/models/model_0100.pt"
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
model.to(device)
model.eval()

import cv2
from tqdm import tqdm

NEW_WIDTH, NEW_HEIGHT = 640, 360
MAX_SIZE = 64

def read(vid, q_out):
    #reads every frame and sends them to the next pipeline stage
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT))
        q_out.put(resized_frame)
    #None is used to mark end of video
    q_out.put(None)
    vid.release()


def process(q_in, q_out):
    #processes each frame through the LaneATT model and draws the results
    while True:
        frame = q_in.get()
        if frame is None:
            q_out.put(None)
            break

        #converts image to tensor and normalize to [0, 1]
        img_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        #run without tracking gradients and decode proposals into lanes
        with torch.no_grad():
            out = model(img_tensor, conf_threshold=0.40, nms_thres=55, nms_topk=4)
            predictions = model.decode(out, as_lanes=True)

        #draws the predicted lanes on a copy of the original frame
        processed_frames = frame.copy()
        if predictions and len(predictions[0]) > 0:
            for lane in predictions[0]:
                pts = lane.points.copy()
                #scale normalized coordinates [0, 1] back to the target resolution
                pts[:, 0] *= NEW_WIDTH
                pts[:, 1] *= NEW_HEIGHT
                pts = pts.round().astype(int)

                #draw each point of the lane in red
                for x, y in pts:
                    cv2.circle(processed_frames, (x, y), 3, (0, 0, 255), thickness=-1)

        q_out.put(processed_frames)

def write(out, q_in, total_frames):
    #writes the processed frames in the output video while showing a status bar
    pbar = tqdm(total=total_frames, desc="Video Processing", unit="frames")
    while True:
        frame = q_in.get()
        if frame is None:
            break
        out.write(frame)
        pbar.update(1)
    out.release()
    pbar.close()

def process_video(input_file, output_file):
    #read video
    vid = cv2.VideoCapture(input_file)

    if not vid.isOpened():
        raise IOError("Unable to open video")

    #get video information
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video length: {total_frames}, fps: {fps}, resolution: {width}x{height}")
    print(f"Output video will be saved to: {os.path.abspath(output_file)}")

    #create the output video
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (NEW_WIDTH, NEW_HEIGHT))

    #pipeline's queues
    queue1 = queue.Queue(maxsize=MAX_SIZE)
    queue2 = queue.Queue(maxsize=MAX_SIZE)

    t1 = Thread(target=read, args=(vid, queue1))
    t2 = Thread(target=process, args=(queue1, queue2))
    t3 = Thread(target=write, args=(out, queue2, total_frames))
    #Start all threads
    t1.start()
    t2.start()
    t3.start()
    #wait for all threads to finish
    t1.join()
    t2.join()
    t3.join()

if __name__ == "__main__":
    path = input("Enter file or folder path: ").strip()
    path = os.path.expanduser(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    #create the empty folder for output videos
    output_dir = 'outputs'
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    valid_extensions = ('.mp4', '.avi', '.mkv', '.mov')

    #check if it's file
    if os.path.isfile(path):
        basename, ext = os.path.splitext(os.path.basename(path))
        output_file = os.path.join(output_dir, f"{basename}_out.mp4")
        process_video(path, output_file)

    #check if it's a directory
    elif os.path.isdir(path):
        #extract all videos
        video_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(valid_extensions)])

        if not video_files:
            raise Exception("No valid video files found")
        else:
            print(f"Found {len(video_files)} video files")
            for video in video_files:
                basename, ext = os.path.splitext(os.path.basename(video))
                output_file = os.path.join(output_dir, f"{basename}_out.mp4")
                process_video(video, output_file)