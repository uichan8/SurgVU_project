# SurgVU_Project

##envs
### conda
```bash
conda create -n SurgVU
conda activate SurgVU
conda install pip
pip install -r requirment.txt
```

## Labeling

### video forwarder server

### labeling server

### 추가 정보
비디오를 다음과 명령어로 이미지로 변환 해야 한다.
    ffmpeg -i Data/clips/video/case_000_00.mp4 -q:v 2 -start_number 0 Data/clips/imgs/case_000_00/%05d.jpg

## tracking