from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
import os
import torch
import numpy as np
import sys
import time
import json
import copy
import boto3
import re
from omegaconf import OmegaConf
import argparse
from pathlib import Path
import uvicorn

# Import MuseTalk modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel

app = FastAPI(title="MuseTalk S3 API", description="API for MuseTalk realtime inference with S3 support")

# Define models for request body
class AudioClip(BaseModel):
    path: str = Field(..., description="Path to the audio file (can be S3 path)")

class AvatarConfig(BaseModel):
    preparation: bool = Field(True, description="Whether to prepare the avatar data")
    video_path: str = Field(..., description="Path to the video file (can be S3 path)")
    bbox_shift: int = Field(0, description="Bounding box shift value")
    audio_clips: Dict[str, str] = Field(..., description="Dictionary of audio clips (can be S3 paths)")

class InferenceConfig(BaseModel):
    version: str = Field("v15", description="Version of MuseTalk: v1 or v15")
    ffmpeg_path: str = Field("./ffmpeg/ffmpeg-master-latest-linux64-gpl/bin/", description="Path to ffmpeg executable")
    gpu_id: int = Field(0, description="GPU ID to use")
    vae_type: str = Field("sd-vae", description="Type of VAE model")
    unet_config: str = Field("./models/musetalk/musetalk.json", description="Path to UNet configuration file")
    unet_model_path: str = Field("./models/musetalk/pytorch_model.bin", description="Path to UNet model weights")
    whisper_dir: str = Field("./models/whisper", description="Directory containing Whisper model")
    bbox_shift: int = Field(0, description="Bounding box shift value")
    result_dir: str = Field('./results', description="Directory for output results")
    extra_margin: int = Field(10, description="Extra margin for face cropping")
    fps: int = Field(25, description="Video frames per second")
    audio_padding_length_left: int = Field(2, description="Left padding length for audio")
    audio_padding_length_right: int = Field(2, description="Right padding length for audio")
    batch_size: int = Field(30, description="Batch size for inference")
    output_vid_name: Optional[str] = Field(None, description="Name of output video file")
    use_saved_coord: bool = Field(False, description="Use saved coordinates to save time")
    saved_coord: bool = Field(False, description="Save coordinates for future use")
    parsing_mode: str = Field('jaw', description="Face blending parsing mode")
    left_cheek_width: int = Field(90, description="Width of left cheek region")
    right_cheek_width: int = Field(90, description="Width of right cheek region")
    skip_save_images: bool = Field(False, description="Whether skip saving images for better generation speed calculation")
    avatars: Dict[str, AvatarConfig] = Field(..., description="Avatar configurations")
    s3_bucket: str = Field("sushant-bucket-mumbai", description="S3 bucket for output upload")
    s3_prefix: str = Field("aws_summit_2025_persona_sync/lip_sync/outputs", description="S3 prefix for output upload")

# Global variables for models
vae = None
unet = None
pe = None
whisper = None
audio_processor = None
fp = None
device = None
timesteps = None
s3_client = None

def fast_check_ffmpeg():
    """Check if ffmpeg is available"""
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

def initialize_models(config):
    """Initialize models with given parameters"""
    global vae, unet, pe, whisper, audio_processor, fp, device, timesteps, s3_client
    
    # Set computing device
    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path=config.unet_model_path,
        vae_type=config.vae_type,
        unet_config=config.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)
    
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    
    # Initialize audio processor and Whisper model
    audio_processor = AudioProcessor(feature_extractor_path=config.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(config.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    
    # Initialize face parser with configurable parameters based on version
    if config.version == "v15":
        fp = FaceParsing(
            left_cheek_width=config.left_cheek_width,
            right_cheek_width=config.right_cheek_width
        )
    else:  # v1
        fp = FaceParsing()
    
    # Initialize S3 client
    s3_client = boto3.client('s3')

def is_s3_path(path):
    """Check if a path is an S3 path"""
    return path.startswith('s3://') or path.startswith('S3://')

def parse_s3_path(s3_path):
    """Parse an S3 path into bucket and key"""
    # Remove s3:// prefix
    path = s3_path.replace('s3://', '').replace('S3://', '')
    # Split into bucket and key
    parts = path.split('/', 1)
    if len(parts) == 2:
        bucket, key = parts
    else:
        bucket = parts[0]
        key = ''
    return bucket, key

def download_from_s3(s3_path, local_dir='./tmp_s3_downloads'):
    """Download a file from S3 to a local path"""
    global s3_client
    
    if not is_s3_path(s3_path):
        return s3_path  # Return original path if not an S3 path
    
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Parse S3 path
    bucket, key = parse_s3_path(s3_path)
    
    # Extract filename from key
    filename = os.path.basename(key)
    local_path = os.path.join(local_dir, filename)
    
    # Download file
    try:
        print(f"Downloading {s3_path} to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        return local_path
    except Exception as e:
        print(f"Error downloading from S3: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download file from S3: {str(e)}")

def upload_to_s3(local_path, bucket, prefix):
    """Upload a file to S3"""
    global s3_client
    
    # Extract filename from local path
    filename = os.path.basename(local_path)
    
    # Create S3 key with prefix
    key = f"{prefix.rstrip('/')}/{filename}"
    
    # Upload file
    try:
        print(f"Uploading {local_path} to s3://{bucket}/{key}")
        s3_client.upload_file(local_path, bucket, key)
        return f"s3://{bucket}/{key}"
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file to S3: {str(e)}")

def extract_filename_without_extension(path):
    """Extract filename without extension from a path"""
    return os.path.splitext(os.path.basename(path))[0]

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup with default parameters"""
    print("Server starting up. Models will be initialized on first request.")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on server shutdown"""
    global vae, unet, pe, whisper, audio_processor, fp
    
    try:
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear model references
        vae = None
        unet = None
        pe = None
        whisper = None
        audio_processor = None
        fp = None
        
        print("Server shutdown: cleaned up resources successfully")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

class Avatar:
    """Avatar class adapted from realtime_inference.py"""
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation, version, result_dir, extra_margin, parsing_mode):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.version = 'v15' if version == 'v15' else 'v1'
        self.extra_margin = extra_margin
        self.parsing_mode = parsing_mode
        
        # Set base path based on version
        self.base_path = f"./intermediate_files/{version}_avatars_{avatar_id}"
        self.avatar_path = self.base_path
        self.video_out_path = f"{result_dir}"
        
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": version
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        
    def init(self):
        """Initialize avatar data"""
        from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
        from musetalk.utils.blending import get_image_prepare_material
        import glob
        import pickle
        import shutil
        import cv2
        
        def osmakedirs(path_list):
            for path in path_list:
                os.makedirs(path) if not os.path.exists(path) else None
                
        def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
            cap = cv2.VideoCapture(vid_path)
            count = 0
            while True:
                if count > cut_frame:
                    break
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
                    count += 1
                else:
                    break
        
        if self.preparation:
            if os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} exists, overwriting it !!")
                response = "y"
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    self.input_latent_list_cycle = torch.load(self.latents_out_path)
                    with open(self.coords_path, 'rb') as f:
                        self.coord_list_cycle = pickle.load(f)
                    input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.frame_list_cycle = read_imgs(input_img_list)
                    with open(self.mask_coords_path, 'rb') as f:
                        self.mask_coords_list_cycle = pickle.load(f)
                    input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.mask_list_cycle = read_imgs(input_mask_list)
            else:
                print("*********************************")
                print(f"  creating avator: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                raise HTTPException(status_code=400, detail=f"{self.avatar_id} does not exist, you should set preparation to True")

            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)

            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                raise HTTPException(status_code=400, detail=f"bbox_shift is changed, you need to re-create it!")
            else:
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_imgs(input_img_list)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_imgs(input_mask_list)
                
    def prepare_material(self):
        """Prepare avatar materials"""
        from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
        from musetalk.utils.blending import get_image_prepare_material
        import glob
        import pickle
        import shutil
        import cv2
        from tqdm import tqdm
        
        def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
            cap = cv2.VideoCapture(vid_path)
            count = 0
            while True:
                if count > cut_frame:
                    break
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
                    count += 1
                else:
                    break
        
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if self.version == "v15":
                y2 = y2 + self.extra_margin
                y2 = min(y2, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]  # Update coord_list bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

            x1, y1, x2, y2 = self.coord_list_cycle[i]
            if self.version == "v15":
                mode = self.parsing_mode
            else:
                mode = "raw"
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)

            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))
        
    def process_frames(self, res_frame_queue, video_len, skip_save_images):
        """Process frames from the queue"""
        import cv2
        import queue
        from musetalk.utils.blending import get_image_blending
        
        print(video_len)
        while True:
            if self.idx >= video_len - 1:
                break
            try:
                start = time.time()
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                continue
            mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            if skip_save_images is False:
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png", combine_frame)
            self.idx = self.idx + 1
            
    def inference(self, audio_path, out_vid_name, fps, skip_save_images, ffmpeg_path, audio_padding_length_left, audio_padding_length_right):
        """Run inference on the avatar with the given audio"""
        import threading
        import queue
        from musetalk.utils.utils import datagen
        import subprocess
        
        # Only create tmp directory if we're saving images
        if not skip_save_images:
            os.makedirs(self.avatar_path + '/tmp', exist_ok=True)
        print("start inference")
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        # Extract audio features
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path, weight_dtype=unet.model.dtype)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            unet.model.dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=audio_padding_length_left,
            audio_padding_length_right=audio_padding_length_right,
        )
        print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        self.idx = 0
        # Create a sub-thread and start it
        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num, skip_save_images))
        process_thread.start()

        gen = datagen(whisper_chunks,
                     self.input_latent_list_cycle,
                     self.batch_size)
        start_time = time.time()
        res_frame_list = []

        for i, (whisper_batch, latent_batch) in enumerate(gen):
            audio_feature_batch = pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)

            pred_latents = unet.model(latent_batch,
                                    timesteps,
                                    encoder_hidden_states=audio_feature_batch).sample
            pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        # Close the queue and sub-thread after all tasks are completed
        process_thread.join()

        if skip_save_images is True:
            print('Total process time of {} frames without saving images = {}s'.format(
                video_num,
                time.time() - start_time))
        else:
            print('Total process time of {} frames including saving images = {}s'.format(
                video_num,
                time.time() - start_time))

        output_path = None
        
        # Make sure the output directory exists
        os.makedirs(self.video_out_path, exist_ok=True)
        
        if out_vid_name is not None:
            if skip_save_images is False:
                # Generate video from saved images
                cmd_img2video = f"{ffmpeg_path}/ffmpeg -y -v warning -r {fps} -f image2 -i {self.avatar_path}/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {self.avatar_path}/temp.mp4"
                print(cmd_img2video)
                subprocess.run(cmd_img2video, shell=True)
                
                output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")
                cmd_combine_audio = f"{ffmpeg_path}/ffmpeg -y -v warning -i {audio_path} -i {self.avatar_path}/temp.mp4 {output_vid}"
                print(cmd_combine_audio)
                subprocess.run(cmd_combine_audio, shell=True)
                
                # Clean up temporary files
                # if os.path.exists(f"{self.avatar_path}/temp.mp4"):
                #     os.remove(f"{self.avatar_path}/temp.mp4")
                # if os.path.exists(f"{self.avatar_path}/tmp"):
                #     import shutil
                #     shutil.rmtree(f"{self.avatar_path}/tmp")
                if os.path.exists(f"./intermediate_files"):
                    import shutil
                    shutil.rmtree(f"./intermediate_files")
                
                print(f"Result is saved to {output_vid}")
                output_path = output_vid
            else:
                # When skip_save_images is True, we can't generate a video from frames
                # but we can create a simple audio-only reference file
                output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")
                
                try:
                    # Create a simple audio-only file
                    cmd_audio_only = f"{ffmpeg_path}/ffmpeg -y -v warning -i {audio_path} -c:a copy {output_vid}"
                    print(cmd_audio_only)
                    subprocess.run(cmd_audio_only, shell=True)
                    print(f"Created audio-only reference file at: {output_vid}")
                    output_path = output_vid
                except Exception as e:
                    print(f"Error creating audio-only reference: {str(e)}")
                    output_path = None
        print("\n")
        return output_path

class InferenceResponse(BaseModel):
    avatar_id: str
    audio_id: str
    output_path: Optional[str] = None
    s3_output_path: Optional[str] = None
    message: str

@app.post("/inference", response_model=InferenceResponse)
async def run_inference(config: InferenceConfig, avatar_id: str, audio_id: str):
    """Run inference with the given configuration"""
    global vae, unet, pe, whisper, audio_processor, fp, s3_client
    try:
        # Check if models are initialized
        if vae is None or unet is None or pe is None or whisper is None or audio_processor is None or fp is None:
            initialize_models(config)
        
        # Configure ffmpeg path
        if not fast_check_ffmpeg():
            print("Adding ffmpeg to PATH")
            # Choose path separator based on operating system
            path_separator = ';' if sys.platform == 'win32' else ':'
            os.environ["PATH"] = f"{config.ffmpeg_path}{path_separator}{os.environ['PATH']}"
            if not fast_check_ffmpeg():
                raise HTTPException(status_code=500, detail="Unable to find ffmpeg, please ensure ffmpeg is properly installed")
        
        # Check if avatar exists in config
        if avatar_id not in config.avatars:
            raise HTTPException(status_code=400, detail=f"Avatar {avatar_id} not found in configuration")
        
        # Check if audio clip exists in avatar config
        avatar_config = config.avatars[avatar_id]
        if audio_id not in avatar_config.audio_clips:
            raise HTTPException(status_code=400, detail=f"Audio {audio_id} not found in avatar {avatar_id} configuration")
        
        # Get audio path and download from S3 if needed
        s3_audio_path = avatar_config.audio_clips[audio_id]
        local_audio_path = download_from_s3(s3_audio_path)
        
        # Get video path and download from S3 if needed
        s3_video_path = avatar_config.video_path
        local_video_path = download_from_s3(s3_video_path)
        
        # Generate output video name if not provided
        output_vid_name = config.output_vid_name
        if output_vid_name is None:
            # Extract filenames without extensions
            video_filename = extract_filename_without_extension(local_video_path)
            audio_filename = extract_filename_without_extension(local_audio_path)
            # Combine filenames
            output_vid_name = f"{video_filename}_{audio_filename}"
        
        # Create avatar instance
        avatar = Avatar(
            avatar_id=avatar_id,
            video_path=local_video_path,
            bbox_shift=avatar_config.bbox_shift if config.version == "v1" else 0,
            batch_size=config.batch_size,
            preparation=avatar_config.preparation,
            version=config.version,
            result_dir=config.result_dir,
            extra_margin=config.extra_margin,
            parsing_mode=config.parsing_mode
        )
        
        # Initialize avatar
        avatar.init()
        
        # Run inference
        output_path = avatar.inference(
            audio_path=local_audio_path,
            out_vid_name=output_vid_name,
            fps=config.fps,
            skip_save_images=config.skip_save_images,
            ffmpeg_path=config.ffmpeg_path,
            audio_padding_length_left=config.audio_padding_length_left,
            audio_padding_length_right=config.audio_padding_length_right
        )
        
        # Upload output to S3 if available
        s3_output_path = None
        if output_path and os.path.exists(output_path):
            s3_output_path = upload_to_s3(output_path, config.s3_bucket, config.s3_prefix)
        
        return InferenceResponse(
            avatar_id=avatar_id,
            audio_id=audio_id,
            output_path=output_path,
            s3_output_path=s3_output_path,
            message="Inference completed successfully"
        )
        
    except Exception as e:
        # Clean up CUDA memory in case of error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
