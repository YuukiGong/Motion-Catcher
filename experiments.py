# from dataclasses import dataclass
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
sys.path.append('./SEARAFT')
sys.path.append('./SEARAFT/core')
import torch
from torch import nn
from torch.nn import functional as F
import argparse
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from models.pipline_condition_svd import StableVideoDiffusionPipeline
from diffusers.models.embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
)
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.models import AutoencoderKLTemporalDecoder
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.unets.unet_3d_blocks import UNetMidBlockSpatioTemporal, get_down_block, get_up_block
from models.addition_net import Optical_ControlNetModel
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from diffusers.utils import load_image, export_to_video,export_to_gif
import SEARAFT


# logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Image path to generate video")
    parser.add_argument("--output_path", type=str, default="./outputs", help="Directory of output")
    parser.add_argument("--video_clip", type=int, default=5, help="Length of synthesized video")
    parser.add_argument("--height", type=int, default=1024, help="Height of synthesized video, and should be a multiple of 32")
    parser.add_argument("--width", type=int, default=576, help="Width of synthesized video, and should be a multiple of 32")
    parser.add_argument("--numframe", type=int, default=14)
    parser.add_argument("--decode_chunk_size", type=int, default=8)
    parser.add_argument("--smoother_steps", nargs='+', default=[19, 20], type=int, help="Timesteps at which using interleaved-frame smoother")
    parser.add_argument("--is_long_video", action='store_true', help="Whether to use hierarchical sampler to produce long video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed of generator")
    parser.add_argument("--version", type=str, default='v10', choices=["v10", "v11"], help="Version of ControlNet")
    parser.add_argument("--frame_rate", type=int, default=None, help="The frame rate of loading input video. Default rate is computed according to video length.")
    parser.add_argument("--temp_video_name", type=str, default=None, help="Default video name")
    
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    device = "cuda:0"
    
    args = get_args()
    pretrained_model_name_or_path = "stabilityai/stable-video-diffusion-img2vid"    
    vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained_model_name_or_path, subfolder="vae", variant="fp16",torch_dtype=torch.float16).to(device)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="image_encoder", variant="fp16",torch_dtype=torch.float16).to(device)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", variant="fp16",torch_dtype=torch.float16).to(device)
    scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path, subfolder="feature_extractor")
    opticallnet = Optical_ControlNetModel.from_pretrained("./wehight",torch_dtype=torch.float16).to(device)

    pipe = StableVideoDiffusionPipeline(vae,image_encoder,unet,scheduler,feature_extractor,opticallnet).to(device)
    # print(pipe.scheduler.compatibles)
    generator = torch.Generator(device="cpu").manual_seed(33)
    opticals = torch.load(".zero_optical.pt").repeat(2, 1, 1, 1, 1).to(torch.float16).to("cuda")
    compute = SEARAFT.Otical_compute()
    # pipe.enable_model_cpu_offload()
    # pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
    # Load the conditioning image
    testdatadir = "./experiments/datasets/t2i_625"
    testdatadir = args.image_path
    add_index = 0
    datasetnames = sorted(os.listdir(testdatadir))[1+add_index:] 
    for index,filename in enumerate(datasetnames):
        index = index + add_index
        imagepath = os.path.join(testdatadir,filename)
        print(imagepath)
        image = load_image(imagepath)
        width = args.width
        height = args.height
        image = image.resize((width, height))
        first_img = image.copy()
        shape = (1,14,4,72,128,)
        allframes = []
        opticals = torch.zeros_like(opticals).to(torch.float16).to("cuda")
        latents = torch.randn(shape, device=device, dtype=torch.float16).to("cuda")
        # addtion_wight = [0.6,0.8,1.0,1.0,1.0]
        last_latent = None
        image_embeddings = None
        with torch.no_grad():
            for clips in range(args.video_clip):
                pipeoutput = pipe(image,last_latent = last_latent, image_embeddings = image_embeddings,decode_chunk_size=args.decode_chunk_size,num_frames = args.numframe,
                                num_inference_steps=25,# generator=generator,
                            optical = opticals,content_image = image, #  addtion_wight = addtion_wight[clips], # latents = latents, #  use_addtionnet = True, min_guidance_scale = 1,max_guidance_scale = 2
                            )
                # print(pipeoutput)
                frames = pipeoutput.frames[0]
                last_latent = pipeoutput.lastlatent
                image_embeddings = pipeoutput.image_embeddings
                allframes.extend(frames[:-1])
                # print(allframes)
                image = frames[-1]
                opticals = compute.compute_optical_list(frames)
                opticals = opticals.unsqueeze(0).repeat(2, 1, 1, 1, 1).to(torch.float16)
                torch.cuda.empty_cache()
        # height=512+64*0,width=512+64*5,
        output_path = args.output_path
        export_to_gif(allframes, os.path.join(output_path,str(index)+".gif") , fps=7)
        print(index,"saved!")
