import os
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import cv2

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def gpu_config(gpu_num):
    # use bfloat16 
    torch.autocast(device_type=f"cuda:{gpu_num}", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(gpu_num).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
    nomal_theta = mask.max() if mask.max() != 0 else 1
    mask_normalized = mask / mask.max()
    
    # 알파 값을 마스크에 적용
    mask_alpha = mask_normalized * alpha
    
    # 블렌딩: 마스크 영역에 노란색 적용
    blended_image = image * (1 - mask_alpha[:, :, np.newaxis]) + color * mask_alpha[:, :, np.newaxis]
    
    # 결과를 8비트 이미지로 변환
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    
    return blended_image

def label_img(img,img_predictor, r=10):
    box = []
    positive_points = []
    negative_points = []
    result_img = img.copy()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    mask = None
    box_exist = False
    drawing = False  # 마우스 클릭 상태 확인
    ix, iy = -1, -1  # 시작 좌표
    bbox = []  # Bounding Box 저장 리스트
    # 마우스 콜백 함수
    def get_point_prompt(event, x, y, flags, param):
        nonlocal positive_points, negative_points, mask, img, result_img, box, box_exist, drawing, ix, iy
        is_pop  = False
        if not box_exist:
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    img_copy = img.copy()
                    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
                    cv2.imshow('img', img_copy)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                box = [ix, iy, x, y]
                box_exist = True
                img_predictor.set_image(img)
                masks, scores, logits = img_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box = np.array(box)[None,:],
                    multimask_output=False,
                )
                mask = masks[0]
                result_img = blend_image_with_mask(img, mask)
                result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                result_img = cv2.rectangle(result_img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
                cv2.imshow('img', result_img)


            return

        else:
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
                    img_predictor.set_image(img)
                    masks, scores, logits = img_predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        box = np.array(box)[None,:],
                        multimask_output=False,
                    )
                    mask = masks[0]
                    result_img = blend_image_with_mask(img, mask)
                else:
                    img_predictor.set_image(img)
                    masks, scores, logits = img_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box = np.array(box)[None,:],
                        multimask_output=False,
                    )
                    mask = masks[0]
                    result_img = blend_image_with_mask(img, mask)
                result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                result_img = cv2.rectangle(result_img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
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
    # mask중 가장 큰 덩어리만 남기고 삭제
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
    return mask

def annotation_process(img_path_list, sam2_checkpoint = "checkpoint/sam2/sam2_hiera_large.pt", model_cfg = "sam2_hiera_l.yaml"):
    gpu_config(1)
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda:1")
    img_predictor = SAM2ImagePredictor(sam2_model)

    result = []
    for p in img_path_list:
        img = Image.open(p)
        img = np.array(img.convert("RGB"))
        mask = label_img(img, img_predictor)
        result.append(mask)

    return result


if __name__ == "__main__":
    result = annotation_process(["image.png"])
    img = Image.open("image.png")
    img = np.array(img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for mask in result:
        img = blend_image_with_mask(img, mask)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
