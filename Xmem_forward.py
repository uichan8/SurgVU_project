import os
import sys
from tools.image_annotation import image_annotation
import cv2
import subprocess
import re

def video_annotation(video_path, output_path = "annotation_result", device = "cuda:0"):
    video_name = video_path.split("/")[-1]
    video_name = video_name.split(".")[0]

    output_path = os.path.join(output_path, video_name)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "annotations"), exist_ok=True)

    image_path = os.path.join(output_path, "images")
    annotation_path = os.path.join(output_path, "annotations")

    command = f"ffmpeg -i {video_path} -map 0:v:0 -c copy -f null - 2>&1"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

   
    output = result.stderr.decode('utf-8')
    match = re.search(r"frame=\s*(\d+)", output)

    if match:
        frame_count = int(match.group(1))
        print(f"Total frames: {frame_count}")
    else:
        frame_count = 0

    print(f"Total frames: {frame_count}")

    #os.system(f"ffmpeg -i {video_path} -q:v 2 -start_number 0 {image_path}/frame_%06d.jpg")

    