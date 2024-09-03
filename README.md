# SurgVU_Project
## clone
```bash
#ssh
git clone --recurse-submodules git@github.com:uichan8/SurgVU_project.git
#https
git clone --recurse-submodules https://github.com/uichan8/SurgVU_project.git
```

## envs
### conda
```bash
conda create -n SurgVU
conda activate SurgVU
conda install pip
pip install -r requirment.txt
```

## model download
### SAM2

### XMem2 (XMem++)
```bash
bash script/XMem2_model_download.sh
```

## Labeling

### video forwarder server

### labeling server

### 추가 정보
비디오를 다음과 명령어로 이미지로 변환 해야 한다.
    ffmpeg -i Data/clips/video/case_000_00.mp4 -q:v 2 -start_number 0 Data/clips/imgs/case_000_00/%05d.jpg

## tracking