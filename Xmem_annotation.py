import os
import sys
from tools.image_annotation import image_annotation
import cv2
import subprocess
import re
import pandas as pd
import numpy as np

valid_class = ['needle driver',#1
               'monopolar curved scissors',#2
               'force bipolar',#3
               'clip applier ',#4
               'cadiere forceps',#5
               'bipolar forceps',#6
               'vessel sealer',#7
               'permanent cautery hook/spatula',#8
               'prograsp forceps',#9
               'stapler',#10
               'grasping retractor',#11
               'tip-up fenestrated grasper',#12
               ]

def video_annotation(video_path, label_path, output_path = "annotation_result", sampling_frame = 400,video_fps = 60, device_num = "0"):
    # 비디오랑, 라벨 가져오고
    video_name = video_path.split("/")[-1]
    video_name = video_name.split(".")[0]
    csv_path = label_path
    video_df = pd.read_csv(csv_path)

    # 출력 경로 생성
    output_path = os.path.join(output_path, video_name)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "annotations"), exist_ok=True)

    image_path = os.path.join(output_path, "images")
    annotation_path = os.path.join(output_path, "annotations")

    # 비디오 프레임 추출
    if len(os.listdir(image_path)) < 100:
        os.system(f"ffmpeg -i {video_path} -q:v 2 -start_number 0 {image_path}/frame_%06d.jpg")

    # 이미지 리스트 생성
    img_list = os.listdir(image_path)
    annotation_list = os.listdir(annotation_path)

    # 라벨 리스트 생성
    class_standard = []
    start_frame_list = []
    for i,row in video_df.iterrows():
        d = []
        fn = int(row['time']*video_fps)
        d.append(fn)
        arm = [row['USM1'],row['USM2'],row['USM3'],row['USM4']]
        for a in arm:
            if a in valid_class:
                d.append(a)
            else:
                d.append(None)
        class_standard.append(d)
        start_frame_list.append(fn)

    #annotation 할 프레임 번호 리스트 생성
    annotation_num_list = [i*sampling_frame for i in range(len(img_list)//sampling_frame)]
    annotation_num_list += start_frame_list
    annotation_num_list = list(set(annotation_num_list))
    annotation_num_list.sort()

    #이미 annotation이 된 경우를 빼고
    annotation_done_list = [int(annotation[6:-4]) for annotation in annotation_list]
    if len(annotation_done_list) == len(annotation_num_list):
        return
    annotation_num_list = [item for item in annotation_num_list if item not in annotation_done_list and item >= 0]
    annotation_num_list = np.array(annotation_num_list)

    #annotation 할 프레임 번호 리스트에 대한 class 리스트 생성
    annotation_class_idx = np.zeros(len(annotation_num_list))
    for i,row in video_df.iterrows():
        if i == 0:
            continue
        annotation_class_idx += (annotation_num_list >= row['time']*video_fps)
    annotation_class_idx = annotation_class_idx.astype(int)

    #class가 None만 있는 경우 annotation 제외
    annotation_class_list = np.array(class_standard)[annotation_class_idx]
    final_annotation_class_idx = []
    for i,acl in enumerate(annotation_class_list):
        if (acl[1:] != None).any():
            final_annotation_class_idx.append(i)
    annotation_num_list = annotation_num_list[final_annotation_class_idx]
    annotation_class_list = annotation_class_list[final_annotation_class_idx]
    
    annotation_img_path_list = [os.path.join(image_path, f"frame_{i:06d}.jpg") for i in annotation_num_list]
    annotation_mask_path_list = [os.path.join(annotation_path, f"frame_{i:06d}.png") for i in annotation_num_list]

    image_annotation(annotation_img_path_list, annotation_mask_path_list, annotation_class_list, device_num = device_num)




    
if __name__ == "__main__":
    root_path = "Data"
    video_path_list = os.listdir(root_path+"/clips")
    video_path_list.sort()
    label_path_list = []
    for video_path in video_path_list:
        label_path_list.append(os.path.join(root_path+"/labels",video_path.split(".")[0]+".csv"))

    output_path = "annotation_result"
    for video_path, label_path in zip(video_path_list, label_path_list):
        video_path = os.path.join("Data/clips", video_path)
        video_annotation(video_path,label_path, output_path, device_num = 0)