import os
import cv2

def imd2vid(imd_path, vid_path):
    img_list = os.listdir(imd_path)
    img_list.sort()
    sample_img = cv2.imread(os.path.join(imd_path, img_list[0]))
    # 동영상 저장 설정
    frame_width = sample_img.shape[1]
    frame_height = sample_img.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱을 XVID에서 mp4v로 변경
    out = cv2.VideoWriter(vid_path, fourcc, 30.0, (frame_width, frame_height))

    for img_name in img_list:
        img = cv2.imread(os.path.join(imd_path, img_name))
        out.write(img)
    out.release()

if __name__ == "__main__":
    imd2vid("Data/clips/result", "output.mp4")