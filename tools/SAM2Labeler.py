import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import threading
import subprocess
import shutil
import cv2
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = '1'

class SAM2Labeler:
    def __init__(self,checkpoint = "checkpoint/sam2/sam2_hiera_large.pt",cfg = "sam2_hiera_l.yaml"):
        self.sam2_checkpoint = checkpoint
        self.model_cfg = cfg
        
        #image predictor
        self.gpu_config(0)
        self.gpu_config(1)

        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device="cuda:1")
        self.img_predictor = SAM2ImagePredictor(self.sam2_model)

        #video predictor
        self.video_predictor = build_sam2_video_predictor(checkpoint=self.sam2_checkpoint, config_file=self.model_cfg, device=f"cuda:0")
    
    @staticmethod
    def gpu_config(gpu_num):
        # use bfloat16 
        torch.autocast(device_type=f"cuda:{gpu_num}", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(gpu_num).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @staticmethod
    def blend_image_with_mask(image, mask, color=np.array([255, 255, 0], dtype=np.float32), alpha=0.5):
        """
        이미지에 노란색 마스크를 적용하여 블렌딩된 이미지를 반환합니다.

        Parameters:
        - image: 원본 이미지 (3채널, RGB)
        - mask: 마스크 이미지 (1채널, grayscale), 값의 범위는 [0, 255]
        - color: 마스크에 적용할 색상 (기본값: 노란색)
        - alpha: 마스크의 투명도 조정 (0.0 ~ 1.0)

        Returns:
        - blended_image: 노란색 마스크가 적용된 이미지 (3채널, RGB)
        """
        # 마스크를 0~1 범위로 정규화
        mask_normalized = mask
        
        # 알파 값을 마스크에 적용
        mask_alpha = mask_normalized * alpha
        
        # 블렌딩: 마스크 영역에 노란색 적용
        blended_image = image * (1 - mask_alpha[:, :, np.newaxis]) + color * mask_alpha[:, :, np.newaxis]
        
        # 결과를 8비트 이미지로 변환
        blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
        
        return blended_image

    # image labeler    
    def label_img_with_points(self, img, r=10,box = []):
        positive_points = []
        negative_points = []
        result_img = img.copy()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        mask = None
        # 마우스 콜백 함수
        def get_point_prompt(event, x, y, flags, param):
            nonlocal positive_points, negative_points,mask, img, result_img, box
            is_pop  = False
            if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭 시
                # 클릭한 위치가 기존의 동그라미 안에 있는지 확인
                for i, (cx, cy) in enumerate(positive_points):
                    if np.sqrt((cx - x) ** 2 + (cy - y) ** 2) <= r:
                        positive_points.pop(i)
                        is_pop = True
                    
                for i, (cx, cy) in enumerate(negative_points):
                    if np.sqrt((cx - x) ** 2 + (cy - y) ** 2) <= r:
                        negative_points.pop(i)
                        is_pop = True
                
                # 컨트롤 키가 눌리면 negative_points에 추가
                if not is_pop:
                    if flags & cv2.EVENT_FLAG_CTRLKEY:
                        negative_points.append((x, y))
                    else:
                        positive_points.append((x, y))
                # 마스크 추가
                input_point = np.array(positive_points + negative_points)
                input_label = np.array([1] * len(positive_points) + [0] * len(negative_points))
                if input_point.shape[0] > 0:
                    self.img_predictor.set_image(img)
                    masks, scores, logits = self.img_predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=False,
                    )
                    mask = masks[0]
                    result_img = self.blend_image_with_mask(img, mask)
                else:
                    result_img = img
                result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                # 포인트 추가
                for x, y in positive_points:
                    cv2.circle(result_img, (x, y), r, (255, 0, 0), -1)
                for x, y in negative_points:
                    cv2.circle(result_img, (x, y), r, (0, 0, 255), -1)

        cv2.namedWindow("img")
        cv2.setMouseCallback("img", get_point_prompt)

        while True:
            cv2.imshow("img", result_img)  # result_img를 보여줌으로써 업데이트된 내용을 반영
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC 키를 누르면 종료
                break

        cv2.destroyAllWindows()
        result_positive_points = positive_points.copy()
        result_negative_points = negative_points.copy()
        positive_points = []
        negative_points = []
        return mask, result_positive_points, result_negative_points
    
    def label_img_with_multi_box(self,img):
        pass
    
    def label_img_with_single_box(self,img):
        pass

    # video labeler
    def label_video_with_points(self,video_dir,frame_batch = 300,visualize = True):
        # make_path
        if not os.path.exists("video_process"):
            os.makedirs("video_process", exist_ok=True)
            os.makedirs("video_process/frame", exist_ok=True)
            os.makedirs("video_process/mask", exist_ok=True)
            os.makedirs("video_process/bbox", exist_ok=True)
        if not os.path.exists("result"):
            os.makedirs("result", exist_ok=True)
        if not os.path.exists(f"result/{video_dir[:-4]}"):
            os.makedirs(f"result/{video_dir[:-4]}", exist_ok=True)
            
        command = [
            'ffmpeg',
            '-i', video_dir,
            '-q:v', '2',
            '-start_number', '0',
            'video_process/frame/%05d.jpg'
        ]
        #subprocess.run(command)

        frame_list = os.listdir("video_process/frame")
        frame_list.sort()
        frame_len = len(frame_list)
        frame_batch_size = frame_len//frame_batch
        
        # for b in range(frame_batch_size+1):
        #     os.makedirs(f"video_process/frame/batch_{str(b+100000)[1:]}", exist_ok=True)
        #     os.makedirs(f"video_process/mask/batch_{str(b+100000)[1:]}", exist_ok=True)
        #     for i,t in enumerate(frame_list[b*frame_batch:(b+1)*frame_batch]):
        #         shutil.move(f"video_process/frame/{t}",f"video_process/frame/batch_{str(b+100000)[1:]}/{str(i+100000)[1:]}.jpg")
        
        shared_data = []
        done_process = 0
        lock = threading.Lock()
        initalization = False
        
        def annotation_thread():
            nonlocal shared_data, lock
            batch_list = os.listdir("video_process/frame")
            batch_list.sort()
            for batch in batch_list:
                b_frame_list = os.listdir(f"video_process/frame/{batch}")
                b_frame_list.sort()
                target = b_frame_list[0]
                img = Image.open(f"video_process/frame/{batch}/{target}")
                img = np.array(img.convert("RGB"))
                _, positive_points, negative_points = self.label_img_with_points(img)
                data = {
                    "batch_id":batch,
                    "positive_points":positive_points,
                    "negative_points":negative_points
                }
                with lock:
                    shared_data.append(data)
        
        def video_processing_thread():
            nonlocal done_process, shared_data, lock
            while True:
                time.sleep(0.5)
                with lock:
                    if len(shared_data) != 0:
                        data = shared_data.pop(0)
                    else:
                        continue
                        
                batch_id = data["batch_id"]
                video_path = os.path.join(f"video_process/frame/{batch_id}")
                ann_obj_id = 1
                
                positive_points = data["positive_points"]
                negative_points = data["negative_points"]
                input_point = np.array(positive_points + negative_points).astype(np.float32)
                input_label = np.array([1] * len(positive_points) + [0] * len(negative_points)).astype(np.int32)
                
                inference_state = self.video_predictor.init_state(video_path=video_path)
                self.video_predictor.reset_state(inference_state)
                self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,   
                    obj_id=ann_obj_id,
                    points=input_point,
                    labels=input_label,
                )
                with lock:
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                        predict_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                        save_path = os.path.join(f"video_process/mask/{batch_id}",f"{str(out_frame_idx+100000)[1:]}.png")
                        cv2.imwrite(save_path,np.squeeze(predict_mask*255).astype(np.uint8))
                del inference_state
                done_process += 1
                print(f"done process : {done_process}/{frame_batch_size}")
                
                if done_process == frame_batch_size:
                    break
                
        thread1 = threading.Thread(target=annotation_thread)
        thread2 = threading.Thread(target=video_processing_thread)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        
        print(1)
                



if __name__ == "__main__":
    labeler = SAM2Labeler()
    labeler.label_video_with_points("Data/clips/video/case_000_00.mp4")
    #labeler.label_img_with_points(cv2.imread("video_process/frame/batch_00000/00000.jpg"))