import requests
import json
import os

# Create the result directory if it doesn't exist
# os.makedirs("./results/client-example-s3", exist_ok=True)

# API endpoint
API_URL = "http://localhost:8080/inference"  # Note the port change to 8080

# Example configuration for S3 API
config = {
    # "version": "v15",
    # "ffmpeg_path": "./ffmpeg/ffmpeg-master-latest-linux64-gpl/bin/",
    # "gpu_id": 0,
    # "vae_type": "sd-vae",
    # "unet_config": "./models/musetalk/musetalk.json",
    # "unet_model_path": "./models/musetalk/pytorch_model.bin",
    # "whisper_dir": "./models/whisper",
    # "bbox_shift": 0,
    # "result_dir": "./results/client-example-s3",  # Local directory for results
    # "extra_margin": 10,
    # "fps": 25,
    # "audio_padding_length_left": 2,
    # "audio_padding_length_right": 2,
    # "batch_size": 35,
    # "output_vid_name": None,  # Set to None to use the combined filename from video and audio
    # "use_saved_coord": False,
    # "saved_coord": False,
    # "parsing_mode": "jaw",
    # "left_cheek_width": 90,
    # "right_cheek_width": 90,
    # "skip_save_images": False,
    # "s3_bucket": "sushant-bucket-mumbai",  # S3 bucket for output upload
    # "s3_prefix": "aws_summit_2025_persona_sync/lip_sync/outputs",  # S3 prefix for output upload
    "avatars": {
        "avator_1": {
            "preparation": True,
            # Example S3 path for video
            "video_path": "s3://sushant-bucket-mumbai/aws_summit_2025_persona_sync/cog_video_generator/3c14f541-121b-42b6-bb7c-21f8b5107f0c/intermediate_video.mp4",
            "audio_clips": {
                # Example S3 path for audio
                # "audio_0": "s3://sushant-bucket-mumbai/aws_summit_2025_persona_sync/seedvc/recreated_audio/remix_vc_9c6bddb6_audio.wav"
                "audio_0": "s3://sushant-bucket-mumbai/aws_summit_2025_persona_sync/seedvc/recreated_audio/remix_vc_dfce4fd5_audio.wav"
            }
        }
    }
}

# Avatar and audio to use
avatar_id = "avator_1"
audio_id = "audio_0"

# Make the API request
try:
    print("Sending request to MuseTalk S3 API...")
    response = requests.post(
        API_URL,
        json=config,
        params={"avatar_id": avatar_id, "audio_id": audio_id}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\nInference completed successfully!")
        print(f"Avatar ID: {result['avatar_id']}")
        print(f"Audio ID: {result['audio_id']}")
        print(f"Local output path: {result['output_path']}")
        print(f"S3 output path: {result['s3_output_path']}")
        print(f"Message: {result['message']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"Request failed: {str(e)}")
