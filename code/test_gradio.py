import sys

import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data.util import read_img 
from data import create_dataloader, create_dataset
from models import create_model
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from utils.JPEG import DiffJPEG
from models.modules.Quantization import Quantization


def img2tensor(img, bgr2rgb=True, device="cpu", add_batch=False):
    """
    Convert numpy image to torch tensor (C,H,W) in [0,1] float32.
    - Accepts (H,W,C) or (H,W) numpy arrays in [0,255] or [0,1].
    - If bgr2rgb=True, assumes input is BGR and converts to RGB safely.
    - add_batch=True will return (1,C,H,W).
    """
    # If someone passes a PIL.Image
    if hasattr(img, "mode"):
        img = np.array(img)

    if img.ndim == 2:
        # (H,W) -> (1,H,W)
        arr = img[np.newaxis, ...]
    elif img.ndim == 3:
        # (H,W,C) -> optionally BGR->RGB without negative strides
        if bgr2rgb and img.shape[2] == 3:
            # Use channel indexing to avoid negative strides:
            # B,G,R -> R,G,B
            img = img[..., [2, 1, 0]]
        # (H,W,C) -> (C,H,W)
        arr = np.transpose(img, (2, 0, 1))
    else:
        raise TypeError(f"Expect 2D or 3D numpy array, got ndim={img.ndim}")

    # Ensure contiguous (fixes negative/odd strides and makes from_numpy happy)
    arr = np.ascontiguousarray(arr)

    # to float32 in [0,1]
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    if arr.max() > 1.0:
        arr = arr / 255.0

    tensor = torch.from_numpy(arr).to(device)

    if add_batch:
        tensor = tensor.unsqueeze(0)  # (1,C,H,W)

    return tensor

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def load_image(image, message = None):
    # img_GT = read_img(None, image_path)
    img_GT = image / 255
    # print(img_GT)
    img_GT = img_GT[:, :, [2, 1, 0]]
    img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float().unsqueeze(0)
    # img_GT = torch.nn.functional.interpolate(img_GT, size=(128, 128), mode='nearest', align_corners=None)
    img_GT = img_GT.unsqueeze(0)

    _, T, C, W, H = img_GT.shape
    list_h = []
    R = 0
    G = 0
    B = 255
    print("W, H in load_image: ", W, H)
    image = Image.new('RGB', (W, H), (R, G, B))
    result = np.array(image) / 255.
    expanded_matrix = np.expand_dims(result, axis=0) 
    expanded_matrix = np.repeat(expanded_matrix, T, axis=0)
    imgs_LQ = torch.from_numpy(np.ascontiguousarray(expanded_matrix)).float()
    imgs_LQ = imgs_LQ.permute(0, 3, 1, 2)
    imgs_LQ = torch.nn.functional.interpolate(imgs_LQ, size=(W, H), mode='nearest', align_corners=None)
    imgs_LQ = imgs_LQ.unsqueeze(0)

    list_h.append(imgs_LQ)

    list_h = torch.stack(list_h, dim=0)

    return {
            'LQ': list_h,
            'GT': img_GT,
            'MES': message
        }


def image_editing(image_numpy, prompt):
    print("====================== EDIT PHASE ===================")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        ).to("cuda")
    
    pil_image = Image.fromarray(image_numpy)
    print(mask_image.shape)
    print("maskmin", mask_image.min(), "maskmax", mask_image.max())
    mask_image = Image.fromarray(mask_image.astype(np.uint8)).convert("L")
    # image_init = pil_image.convert("RGB").resize((512, 512))
    
    h, w = mask_image.size
    
    image_inpaint = pipe(prompt=prompt, image=image_init, mask_image=mask_image, height=w, width=h).images[0]
    image_inpaint = np.array(image_inpaint) / 255.
    image = np.array(image_init) / 255.
    mask_image = np.array(mask_image)
    mask_image = np.stack([mask_image] * 3, axis=-1) / 255.
    mask_image = mask_image.astype(np.uint8)
    image_fuse = image * (1 - mask_image) + image_inpaint * mask_image

    return image_fuse

def image_editing_tung(image_numpy, y_forw, model_index, quality=70):
    with torch.no_grad():
        if (model_index == 0):
            # img = image_numpy.copy()
            # if img.ndim == 2:
            #     img = np.stack([img]*3, axis=-1)
            # # ensure float32 in [0,1]
            # if img.dtype != np.float32:
            #     img = img.astype(np.float32)
            # if img.max() > 1.0:
            #     img = img / 255.0
            # print("SHAPE of img:", img.shape)
            # # create a batched tensor on CUDA with shape (1, C, H, W)
            # # set bgr2rgb=False assuming image_numpy is RGB coming from Gradio/PIL
            # tensor = img2tensor(img, bgr2rgb=True, device="cuda", add_batch=True)

            # diffjpeg = DiffJPEG(differentiable=True, quality=int(quality)).to("cuda")
            # with torch.no_grad():
            #     out = diffjpeg(tensor)

            # result = torch.clamp(out,0,1)

            # result_np = util.tensor2img(result)
            # return result_np
            NL = quality
            diffJPEG = DiffJPEG(differentiable=True, quality=int(NL)).cuda()
            y_forw = diffJPEG(y_forw)
            result = torch.clamp(y_forw,0,1)

            lr_img = util.tensor2img(result)

    quantization = Quantization()
    y = quantization(y_forw)
    
    return lr_img, y_forw, y