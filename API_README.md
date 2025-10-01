# MuseTalk FastAPI Server

This is a FastAPI-based server for MuseTalk's realtime inference. It allows you to run lip-sync inference through a REST API.

## Installation

### Prerequisites

Make sure you have already installed the MuseTalk dependencies:

```bash
conda create -n MuseTalk python==3.10
conda activate MuseTalk

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"
```

### API Server Dependencies

Configure AWS CLI for S3 Access.

Install the additional dependencies required for the API server:

```bash
pip install -r requirements_realtime_api.txt
```

## Running the Server

Start the API server with:

```bash
python api.py
```

By default, the server will run on `http://localhost:8800`.

## API Documentation

Once the server is running, you can access the API documentation at:

- Swagger UI: `http://localhost:8800/docs`
- ReDoc: `http://localhost:8800/redoc`

## API Endpoints

### POST /inference

Run inference with the given configuration.

**Parameters:**

- `config` (body): The inference configuration
- `avatar_id` (query): The ID of the avatar to use
- `audio_id` (query): The ID of the audio clip to use

**Example Request:**

```python
import requests
import json

API_URL = "http://localhost:8800/inference"

config = {
    "version": "v15",
    "ffmpeg_path": "./ffmpeg-4.4-amd64-static/",
    "gpu_id": 0,
    "vae_type": "sd-vae",
    "unet_config": "./models/musetalk/musetalk.json",
    "unet_model_path": "./models/musetalk/pytorch_model.bin",
    "whisper_dir": "./models/whisper",
    "bbox_shift": 0,
    "result_dir": "./results",
    "extra_margin": 10,
    "fps": 25,
    "audio_padding_length_left": 2,
    "audio_padding_length_right": 2,
    "batch_size": 20,
    "output_vid_name": None,
    "use_saved_coord": False,
    "saved_coord": False,
    "parsing_mode": "jaw",
    "left_cheek_width": 90,
    "right_cheek_width": 90,
    "skip_save_images": False,
    "avatars": {
        "avator_1": {
            "preparation": True,
            "video_path": "data/video/yongen.mp4",
            "bbox_shift": 5,
            "audio_clips": {
                "audio_0": "data/audio/yongen.wav",
                "audio_1": "data/audio/eng.wav"
            }
        }
    }
}

avatar_id = "avator_1"
audio_id = "audio_0"

response = requests.post(
    API_URL,
    json=config,
    params={"avatar_id": avatar_id, "audio_id": audio_id}
)

if response.status_code == 200:
    result = response.json()
    print(f"Output path: {result['output_path']}")
```

**Response:**

```json
{
  "avatar_id": "avator_1",
  "audio_id": "audio_0",
  "output_path": "./results/v15/avatars/avator_1/vid_output/audio_0.mp4",
  "message": "Inference completed successfully"
}
```

## Configuration Format

The API accepts the same configuration parameters as the original `realtime_inference.py` script, but in JSON format instead of YAML. Here's an example configuration:

```json
{
  "version": "v15",
  "ffmpeg_path": "./ffmpeg-4.4-amd64-static/",
  "gpu_id": 0,
  "vae_type": "sd-vae",
  "unet_config": "./models/musetalk/musetalk.json",
  "unet_model_path": "./models/musetalk/pytorch_model.bin",
  "whisper_dir": "./models/whisper",
  "bbox_shift": 0,
  "result_dir": "./results",
  "extra_margin": 10,
  "fps": 25,
  "audio_padding_length_left": 2,
  "audio_padding_length_right": 2,
  "batch_size": 20,
  "output_vid_name": null,
  "use_saved_coord": false,
  "saved_coord": false,
  "parsing_mode": "jaw",
  "left_cheek_width": 90,
  "right_cheek_width": 90,
  "skip_save_images": false,
  "avatars": {
    "avator_1": {
      "preparation": true,
      "video_path": "data/video/yongen.mp4",
      "bbox_shift": 5,
      "audio_clips": {
        "audio_0": "data/audio/yongen.wav",
        "audio_1": "data/audio/eng.wav"
      }
    }
  }
}
```

## Example Client

A simple example client is provided in `client_example.py`. You can run it with:

```bash
python client_example.py
```

## Notes

- The server initializes models on the first request, which may take some time.
- Models are kept in memory between requests for faster inference.
- The server cleans up resources on shutdown.
