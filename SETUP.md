# Purpose

Following are the instructions that were leveraged to figure out the model and developing the serving API. If you cloned this repo then you already have the code that work with the serving files. Just make sure to download the weights.

# Instance: 
    - g6e.4xlarge
# AMI:
    - Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20250205
    - Supported EC2 instances: G4dn, G5, G6, Gr6, G6e, P4d, P4de, P5, P5e, P5en. Release notes: https://aws.amazon.com/releasenotes/aws-deep-learning-base-gpu-ami-ubuntu-22-04/
# Conda

If conda not present:
```sh
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
```

If/Once present

```sh
conda init --all
```

# HuggingFace


```sh
conda install -c conda-forge huggingface_hub[cli]

huggingface-cli login --token HF_TOKEN 
```


# MuseTalk

```
mkdir lip_sync
cd lip_sync
git clone https://github.com/TMElyralab/MuseTalk.git
```

Note: Git Clone was done with [changes from 12 April 2025](https://github.com/TMElyralab/MuseTalk/tree/67e7ee3c7397bcfd03e123398e5497f31be1bf92), the serving file might not work 

Setup as per the read me in the github repo.

```sh
cd MuseTalk
conda create -n MuseTalk python==3.10
conda activate MuseTalk

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install boto3
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"


mkdir ffmpeg
cd ffmpeg

wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz
tar -xvJf ffmpeg-master-latest-linux64-gpl.tar.xz
export FFMPEG_PATH=/home/ubuntu/lip_sync/MuseTalk/ffmpeg/ffmpeg-master-latest-linux64-gpl/bin  # add this to bashrc
export PATH=$FFMPEG_PATH:$PATH   # add this to bashrc

cd /home/ubuntu/lip_sync/MuseTalk/


chmod +x ./download_weights.sh

sh ./download_weights.sh   ## run the commands individually after cleaning up models folder if this fails


# sh inference.sh v1.5 realtime

python -m scripts.inference --inference_config ./configs/inference/sample.yaml --result_dir ./results/sample --unet_model_path ./models/musetalkV15/unet.pth --unet_config ./models/musetalkV15/musetalk.json --version v15 --ffmpeg_path ./ffmpeg/ffmpeg-master-latest-linux64-gpl/bin/


python -m scripts.inference --inference_config ./configs/inference/sample-2.yaml --result_dir ./results/sample-2 --unet_model_path ./models/musetalkV15/unet.pth --unet_config ./models/musetalkV15/musetalk.json --version v15 --ffmpeg_path ./ffmpeg/ffmpeg-master-latest-linux64-gpl/bin/

python -m scripts.realtime_inference --inference_config ./configs/inference/sample-2-realtime.yaml --result_dir ./results/sample-2 --unet_model_path ./models/musetalkV15/unet.pth --unet_config ./models/musetalkV15/musetalk.json --version v15 --ffmpeg_path ./ffmpeg/ffmpeg-master-latest-linux64-gpl/bin/ --skip_save_images --fps 25

python -m scripts.realtime_inference --inference_config ./configs/inference/aira-realtime.yaml --result_dir ./results/aira --unet_model_path ./models/musetalkV15/unet.pth --unet_config ./models/musetalkV15/musetalk.json --version v15 --ffmpeg_path ./ffmpeg/ffmpeg-master-latest-linux64-gpl/bin/ --fps 25
```

## Server Setup and Use

Details are given in 
1. [MuseTalk Serving API Documentation that stores files locally](MuseTalk/API_README.md)
2. [MuseTalk Serving API Documentation that stores files in S3](MuseTalk/API_README_S3.md)