import os
import sys
from tools.image_annotation import image_annotation
import cv2

sys.path.append("lib/XMem2")
from inference.run_on_video import select_k_next_best_annotation_candidates
from inference.run_on_video import run_on_video



def video_annotation(video_path, output_path):
    video_name = video_path.split("/")[-1]
    video_name = video_name.split(".")[0]

    output_path = os.path.join(output_path, video_name)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "annotations"), exist_ok=True)

    image_path = os.path.join(output_path, "images")
    mask_path = os.path.join(output_path, "masks")
    annotation_path = os.path.join(output_path, "annotations")
    # video to images
    # os.system(f"ffmpeg -i {video_path} -q:v 2 -start_number 0 {image_path}/frame_%06d.jpg")

    sampling_rate = 400
    annotation_num_list = [i*sampling_rate for i in range(len(os.listdir(image_path))//sampling_rate)]
    annotation_img_path_list = [os.path.join(image_path, f"frame_{i:06d}.jpg") for i in annotation_num_list]
    annotation_mask_path_list = [os.path.join(annotation_path, f"frame_{i:06d}.png") for i in annotation_num_list]

    # first frame annotation
    masks = image_annotation(annotation_img_path_list,color='red')
    for i, mask in enumerate(masks):
        cv2.imwrite(annotation_mask_path_list[i], mask)

    # video annotation
    run_on_video(image_path, annotation_path, output_path, annotation_num_list)






if __name__ == "__main__":
    video_path = "Data/clips/case_000_00.mp4"
    output_path = "annotation_result"
    video_annotation(video_path, output_path)