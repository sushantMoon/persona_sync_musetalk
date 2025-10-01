import requests
import json
import os

# Create the result directory if it doesn't exist
# os.makedirs("./results/client-example", exist_ok=True)

# API endpoint
API_URL = "http://localhost:8080/inference"

# Example configuration based on realtime.yaml
# config = {
#     "version": "v15",
#     "ffmpeg_path": "./ffmpeg/ffmpeg-master-latest-linux64-gpl/bin/",
#     "gpu_id": 0,
#     "vae_type": "sd-vae",
#     "unet_config": "./models/musetalk/musetalk.json",
#     "unet_model_path": "./models/musetalk/pytorch_model.bin",
#     "whisper_dir": "./models/whisper",
#     "bbox_shift": 0,
#     "result_dir": "./results/client-example", # can change but do create the base folder
#     "extra_margin": 10,
#     "fps": 25,
#     "audio_padding_length_left": 2,
#     "audio_padding_length_right": 2,
#     # "batch_size": 20,
#     "batch_size": 35,
#     "output_vid_name": "client_sample_video", # If None then Uses audio_id as the output video name
#     "use_saved_coord": False,
#     "saved_coord": False,
#     "parsing_mode": "jaw",
#     "left_cheek_width": 90,
#     "right_cheek_width": 90,
#     "skip_save_images": False, # if we need the output file and remove the intermediate files
#     "avatars": {
#         "avator_1": {
#             "preparation": True,
#             "video_path" : "/home/ubuntu/lip_sync/MuseTalk/sample_input/video/video_afe6e7e9-8192-41e5-b3f5-1249c78bdef5.mp4",
#             # "bbox_shift": 5,
#             "audio_clips": {
#                 "audio_0": "/home/ubuntu/lip_sync/MuseTalk/sample_input/audio/Adele_Hello.wav"
#             }
#         }
#     }
# }

config = {
    # "version": "v15",
    # "ffmpeg_path": "./ffmpeg/ffmpeg-master-latest-linux64-gpl/bin/",
    # "gpu_id": 0,
    # "vae_type": "sd-vae",
    # "unet_config": "./models/musetalk/musetalk.json",
    # "unet_model_path": "./models/musetalk/pytorch_model.bin",
    # "whisper_dir": "./models/whisper",
    # "bbox_shift": 0,
    # "result_dir": "./results/client-example", # can change but do create the base folder
    # "extra_margin": 10,
    # "fps": 25,
    # "audio_padding_length_left": 2,
    # "audio_padding_length_right": 2,
    # # "batch_size": 20,
    # "batch_size": 35,
    "output_vid_name": "client_sample_video", # If None then Uses audio_id as the output video name
    # "use_saved_coord": False,
    # "saved_coord": False,
    # "parsing_mode": "jaw",
    # "left_cheek_width": 90,
    # "right_cheek_width": 90,
    # "skip_save_images": False, # if we need the output file and remove the intermediate files
    "avatars": {
        "avator_1": {
            "preparation": True,
            "video_path" : "/home/ubuntu/lip_sync/MuseTalk/sample_input/video/video_afe6e7e9-8192-41e5-b3f5-1249c78bdef5.mp4",
            # "bbox_shift": 5,
            "audio_clips": {
                "audio_0": "/home/ubuntu/lip_sync/MuseTalk/sample_input/audio/Adele_Hello.wav"
            }
        }
    }
}

# Avatar and audio to use
avatar_id = "avator_1"
audio_id = "audio_0"

# Make the API request
try:
    response = requests.post(
        API_URL,
        json=config,
        params={"avatar_id": avatar_id, "audio_id": audio_id}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("Inference completed successfully!")
        print(f"Avatar ID: {result['avatar_id']}")
        print(f"Audio ID: {result['audio_id']}")
        print(f"Output path: {result['output_path']}")
        print(f"Message: {result['message']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"Request failed: {str(e)}")
