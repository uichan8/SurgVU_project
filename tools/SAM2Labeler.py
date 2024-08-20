import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import threading
import cv2

class SAM2Labeler:
    def __init__(self,checkpoint = "checkpoint/sam2/sam2_hiera_tiny.pt",cfg = "sam2_hiera_t.yaml"):
        self.sam2_checkpoint = checkpoint
        self.model_cfg = cfg
        
        #image predictor
        self.gpu_config(0)
        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device="cuda:0")
        self.img_predictor = SAM2ImagePredictor(self.sam2_model)

        #video predictor
        self.gpu_config(1)
        self.video_predictor = build_sam2_video_predictor(checkpoint=self.sam2_checkpoint, config_file=self.model_cfg, device=f"cuda:1")
    
    @staticmethod
    def gpu_config(gpu_num):
        # use bfloat16
        torch.autocast(device_type=f"cuda:{gpu_num}", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(gpu_num).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @staticmethod
    def blend_image_with_mask(image, mask):
        """
        이미지에 노란색 마스크를 적용하여 블렌딩된 이미지를 반환합니다.

        Parameters:
        - image: 원본 이미지 (3채널, RGB)
        - mask: 마스크 이미지 (1채널, grayscale), 값의 범위는 [0, 255]

        Returns:
        - blended_image: 노란색 마스크가 적용된 이미지 (3채널, RGB)
        """
        # 마스크를 0~1 범위로 정규화
        mask_normalized = mask / 1
        
        # 노란색 (R=255, G=255, B=0)
        yellow = np.array([255, 255, 0], dtype=np.float32)
        
        # 블렌딩: 마스크 영역에 노란색 적용
        blended_image = image * (1 - mask_normalized[:, :, np.newaxis]) + yellow * mask_normalized[:, :, np.newaxis]
        
        # 결과를 8비트 이미지로 변환
        blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
        
        return blended_image

    # image labeler    
    def label_img_with_points(self, img, r=10, positive_points = [], negative_points = [],box = []):
        result_img = img.copy()
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
        return mask, positive_points, negative_points
    
    def label_img_with_multi_box(self,img):
        pass
    
    def label_img_with_single_box(self,img):
        pass

    # video labeler
    def label_video(self,video_dir,points,labels):
        inference_state = inference_state = predictor.init_state(video_path=video_dir,offload_state_to_cpu=True,offload_video_to_cpu=True,async_loading_frames=True)
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        self.predictor.reset_state(inference_state)



if __name__ == "__main__":
    labeler = SAM2Labeler()
    img = cv2.imread("image.png")
    labeler.label_img(img)