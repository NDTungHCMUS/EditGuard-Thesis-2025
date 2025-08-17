import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
import requests
from copy import deepcopy
import cv2
from test_gradio import load_image, image_editing, img2tensor, image_editing_tung
from utils.my_util import bit_string_to_messagenp, combine_torch_tensors_4d, split_torch_tensors_4d, tensor_to_binary_string, calculate, compute_bit_error_and_accuracy
from utils.my_util_2 import bit_accuracy, split_bits_30, encode_ascii, decode_ascii, split_into_tiles_128
from utils.hamming_code_7_4_new import encode_hamming74, decode_hamming74, recover_30_from_codeword60, parity_30_from_30

from models.modules.Quantization import Quantization
import options.options as option
from utils.JPEG import DiffJPEG
from scipy.io.wavfile import read as wav_read
from scipy.io import wavfile

import os
import math
import argparse
import random
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler
from utils import util
from data.util import read_img
from data import create_dataloader, create_dataset
from models import create_model as create_model_editguard
from diffusers import StableDiffusionInpaintPipeline
import base64
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import hashlib

def sha256_bitstring(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).digest()  # 32 bytes
    return ''.join(f'{b:08b}' for b in digest)   

def text_to_bitstring(text: str, encoding: str = "utf-8") -> str:
    """Encode text -> '0'/'1' bitstring (MSB-first, 8 bit/byte)."""
    return ''.join(f'{b:08b}' for b in text.encode(encoding))

def img_to_base64(filepath):
    with open(filepath, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


logo_base64 = img_to_base64("../logo.png")

html_content = f"""
<div style='display: flex; align-items: center; justify-content: center; padding: 20px;'>
    <img src='data:image/png;base64,{logo_base64}' alt='Logo' style='height: 50px; margin-right: 20px;'>
    <strong><font size='8'>InnoGuard</font></strong>
</div>
"""

# ---------- VN Start ------------------
def hiding2(image_input, text_input, metadata_input, type_correction_code, model):
    bit_input = sha256_bitstring(text_input)
    copyright_list, copyright_padding = split_bits_30(bit_input)
    metadata_input = encode_ascii(metadata_input)
    metadata_list, metadata_padding = split_bits_30(metadata_input)

    tiles_128, coords, orig_hw, padded_hw = split_into_tiles_128(image_input, pad_mode="edge")
    num_child_images = len(tiles_128)
    list_container = []
    list_messageTensor = []
    H, W, C = image_input.shape
    num_child_on_width_size, num_child_on_height_size = H//128, W//128

    for i in range(0, num_child_images):
        if (i < 9):
            message = copyright_list[i]
        elif (i < 9 + len(metadata_list)):
            message = metadata_list[i - 9]
        else:
            if (type_correction_code == 0):
                    message = -1
            elif (type_correction_code != 0):
                if (i < 2 * 9 + len(metadata_list)):
                    message = parity_30_from_30(copyright_list[i - 9 - len(metadata_list)])
                elif (i < 2 * 9 + 2 * len(metadata_list)):
                    message = parity_30_from_30(metadata_list[i - 2 * 9 - len(metadata_list)])
                else :
                    message = -1
    
        if message != -1:
            messagenp = bit_string_to_messagenp(message, batch_size=1)
            message = torch.Tensor(messagenp)
            val_data = load_image(tiles_128[i], message)
            model.feed_data(val_data)
            container, y_forw_res = model.image_hiding()
            list_container.append(y_forw_res)
            list_messageTensor.append(message)
        else:
            messagenp = bit_string_to_messagenp("0" * 30, batch_size=1)
            message = torch.Tensor(messagenp)
            val_data = load_image(tiles_128[i], message)
            model.feed_data(val_data)
            container, y_forw_res = model.image_hiding(embedMessage = False)
            list_container.append(y_forw_res)
            list_messageTensor.append(message)

    parent_container = combine_torch_tensors_4d(list_container, num_child_on_width_size, num_child_on_height_size)
    result = torch.clamp(parent_container,0,1)

    lr_img = util.tensor2img(result)
    return lr_img, lr_img, parent_container, list_messageTensor, copyright_list, metadata_list, copyright_padding, metadata_padding
# ----------------------- VN End ------------------------

import random, secrets, string
# ---------- VN Start ------------------
def rand_text():
    length = random.randint(10, 50)
    # printable ASCII without control chars; allow space
    alphabet = ''.join(chr(i) for i in range(32, 127))  # 32..126
    return ''.join(secrets.choice(alphabet) for _ in range(length))
# ---------- VN End --------------------

# ----------- VN Start -------------
def ImageEdit(img, y_forw, model_index):
    # image, mask = img["image"], img_mask["image"]
    received_image, y_forw_res, y_res = image_editing_tung(img, y_forw, model_index)
    print("============================ End ImageEdit ==========================")
    return received_image, received_image, received_image, y_forw_res, y_res
# ---------- VN End ---------------

def imgae_model_select(ckp_index=0):
    # options
    opt = option.parse("options/test_editguard.yml", is_train=True)
    # distributed training settings
    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id)
        )
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    torch.backends.cudnn.benchmark = True

    # create model
    model = create_model_editguard(opt)

    if ckp_index == 0:
        model_pth = '../checkpoints/16000_G.pth'
    print(model_pth)
    model.load_test(model_pth)
    return model

# ----------- VN Start -----------
def image_to_y_forw(image, device="cuda", bgr2rgb=False, quantize=False):
    """
    Convert a PIL.Image or HWC numpy array -> y_forw-like torch tensor.
    Returns (y_forw, y_quantized_or_None)
    - y_forw: torch.Tensor shape (1,C,H,W), float32 in [0,1] on `device`
    - y_quantized_or_None: Quantization(y_forw) if quantize=True else None
    """
    # reuse img2tensor already in this file to get correct transposes/contiguity
    tensor = img2tensor(image, bgr2rgb=bgr2rgb, device=device, add_batch=True)  # -> (1,C,H,W)
    y_forw = tensor.float().to(device)

    y_q = None
    if quantize:
        quant = Quantization().to(device)
        with torch.no_grad():
            y_q = quant(y_forw)

    return y_forw, y_q
# ----------- VN End -------------

# ------------ VN Start -----------
def revealing2(image_edited, parent_y_forw, parent_y, list_copyright, list_metadata, list_messageTensor, copyright_padding, metadata_padding, type_correction_code, model, upload_separate = False):
    H, W, C = image_edited.shape
    num_child_on_width_size, num_child_on_height_size = H//128, W//128
    num_child_images = num_child_on_width_size * num_child_on_height_size

    if (upload_separate):
        parent_y_forw, parent_y = image_to_y_forw(image_edited, device="cuda", bgr2rgb=True, quantize=True)
    list_rec = split_torch_tensors_4d(parent_y_forw, num_child_on_width_size, num_child_on_height_size)
    list_rec_quantize = split_torch_tensors_4d(parent_y, num_child_on_width_size, num_child_on_height_size)
    
    list_recmessage = []
    list_message = []

    for i in range(0, num_child_images):
        if (type_correction_code == 0 and i < 9 + len(list_metadata)):
            recmessage, message = model.extract(list_messageTensor[i], y_forw = list_rec[i], y = list_rec_quantize[i])
            list_recmessage.append(recmessage)
            list_message.append(message)
        elif (type_correction_code != 0 and i < 2 * 9 + 2 * len(list_metadata)):
            recmessage, message = model.extract(list_messageTensor[i], y_forw = list_rec[i], y = list_rec_quantize[i])
            list_recmessage.append(recmessage)
            list_message.append(message)

    for i in range(0, len(list_message)):
          list_message[i] = tensor_to_binary_string(list_message[i])
          list_recmessage[i] = tensor_to_binary_string(list_recmessage[i])

    copyright_before, copyright_after, copyright_after_ECC, metadata_before, metadata_after, metadata_after_ECC = calculate(list_message, list_recmessage, copyright_blocks=9, metadata_blocks=len(list_metadata), copyright_padding=copyright_padding, metadata_padding=metadata_padding, type_correction_code=type_correction_code)

    bit_error, bit_accuracy = compute_bit_error_and_accuracy(copyright_before, copyright_after, metadata_before, metadata_after)
    bit_error_ECC, bit_accuracy_ECC = compute_bit_error_and_accuracy(copyright_before, copyright_after_ECC, metadata_before, metadata_after_ECC)

    data_after = copyright_after + metadata_after
    data_after_ECC = copyright_after_ECC + metadata_after_ECC

    metadata_after_text_format = decode_ascii("".join(metadata_after))
    metadata_after_ECC_text_format = decode_ascii("".join(metadata_after_ECC))
    return data_after, data_after_ECC, bit_accuracy, bit_accuracy_ECC, metadata_after_text_format, metadata_after_ECC_text_format
# ------------ VN End -------------

def calculate_similarity_percentage(str1, str2):
    if len(str1) == 0:
        return "Original watermark unknown"
    elif len(str1) != len(str2):
        return "Input and output watermark lengths differ"
    total_length = len(str1)
    same_count = sum(1 for x, y in zip(str1, str2) if x == y)
    similarity_percentage = (same_count / total_length) * 100
    return f"{similarity_percentage}%"


# Description
title = "<center><strong><font size='8'>InnoGuard</font></strong></center>"

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

with gr.Blocks(css=css, title="InnoGuard") as demo:
    gr.HTML(html_content)
    model = gr.State(value=None)
    save_h = gr.State(value=None)
    save_w = gr.State(value=None)
    sam_global_points = gr.State([])
    sam_global_point_label = gr.State([])
    sam_original_image = gr.State(value=None)
    sam_mask = gr.State(value=None)

    # ------------------ VN Start ----------------
    y_forw = gr.State(value=None)
    y = gr.State(value = None) # y_forw after quantization
    list_copyright = gr.State(value = None) # List of 30-bit message copyright
    list_metadata = gr.State(value = None) # List of 30-bit message metadata
    list_messageTensor = gr.State(value = []) # List of all tensor of original message
    type_correction_code = gr.State(value = 1) # Keep it
    copyright_padding = gr.State(value = 14) # Padding of last block for copyright
    metadata_padding = gr.State(value = None) # Padding of last block for metadata 
    metadata_after_text_format = gr.State(value = None) # Recovered metadata
    metadata_after_ECC_text_format = gr.State(value = None) # Recovered metadata (with ECC)
     # ------------------ VN End ----------------

    with gr.Tabs():
        with gr.TabItem('Multifunctional Forensic Watermark'):
            DESCRIPTION = """
            ## How to use
            - Upload an image and a copyright watermark (64-bit bitstring), then click **"Embed watermark"** to create a watermarked image.
            - Brush over the regions you want to edit and apply an inpainting algorithm to edit the image.
            - Click **"Extract"** to detect the edited region and recover the embedded watermark.
            """
            gr.Markdown(DESCRIPTION)

            save_inpainted_image = gr.State(value=None)

            with gr.Column():
                with gr.Row():
                    model_list = gr.Dropdown(label="Select model", choices=["LightGuard"], type='index', value = 0)
                    clear_button = gr.Button("Clear all")
                with gr.Group():
                    gr.Markdown("# 1. Embed copyright and metadata into image")
                    with gr.Group():
                        with gr.Column():
                            image_input = gr.Image(
                                label="Original image",
                                interactive=True,
                                type="numpy",
                                value=default_example[0]
                            )
                            with gr.Row():
                                copyright_input = gr.Textbox(
                                    label="Enter copyright watermark (text format)",
                                    placeholder="Type here..."
                                )
                                metadata_input = gr.Textbox(
                                    label="Enter metadata (text format)",
                                    placeholder="Type here..."
                                )
                                rand_copyright = gr.Button("🎲 Randomize copyright")
                                rand_metadata = gr.Button("🎲 Randomize metadata")
                            hiding_button = gr.Button("Embed watermark")
                        with gr.Column():
                            image_watermark = gr.Image(
                                label="Watermarked image",
                                interactive=True,
                                type="numpy"
                            )

                with gr.Group():
                    gr.Markdown("# 2. Tamper image")
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                image_edit = gr.Image(
                                    label="Container Image",
                                    interactive=True,
                                    type="numpy"
                                )
                            inpainting_model_list = gr.Dropdown(
                                label="Choose degradation type",
                                choices=["JPEG Compression (Q = 70)", "Gaussian Noise (σ = 10)", "Stable Diffusion Inpaint", "Adversarial Attacks"],
                                type='index'
                            )
                            inpainting_button = gr.Button("Edit image")
                        with gr.Column():
                            image_edited = gr.Image(     
                                label="Attacked image",
                                interactive=True,
                                type="numpy"
                            )

                with gr.Group():
                    gr.Markdown("# 3. Extract copyright + metadata")
                    with gr.Row():
                        with gr.Column():
                            image_edited_1 = gr.Image(
                                label="Image to extract from",
                                interactive=True,
                                type="numpy"
                            )
                            revealing_button = gr.Button("Extract")
                        with gr.Column():
                            data_after = gr.Textbox(label="Data after")
                            data_after_ECC = gr.Textbox(label = "Data after with ECC")
                            metadata_after_text_format = gr.Textbox(label="Metadata after (text format)")
                            metadata_after_ECC_text_format = gr.Textbox(label="Metadata after (text format) with ECC")
                            acc_output = gr.Textbox(label="Accuracy (without ECC)")
                            acc_ECC_output = gr.Textbox(label="Accuracy (with ECC)")

                model_list.change(
                    imgae_model_select, inputs=[model_list], outputs=[model]
                )
                hiding_button.click(
                    hiding2, inputs=[image_input, copyright_input, metadata_input, type_correction_code, model], outputs=[image_watermark, image_edit, y_forw, list_messageTensor, list_copyright, list_metadata, copyright_padding, metadata_padding]
                )
                rand_copyright.click(
                    rand_text, inputs=[], outputs=[copyright_input]
                )
                rand_metadata.click(
                    rand_text, inputs=[], outputs=[metadata_input]
                )
                inpainting_button.click(
                    ImageEdit,
                    inputs=[image_edit, y_forw, inpainting_model_list],
                    outputs=[image_edited, image_edited_1, save_inpainted_image, y_forw, y]
                )
                revealing_button.click(
                    revealing2,
                    inputs=[image_edited_1, y_forw, y, list_copyright, list_metadata, list_messageTensor, copyright_padding, metadata_padding, type_correction_code, model],
                    outputs=[data_after, data_after_ECC, acc_output, acc_ECC_output, metadata_after_text_format, metadata_after_ECC_text_format]
                )
    demo.load(imgae_model_select, inputs = [gr.State(0)], outputs = [model])
demo.launch(server_name="0.0.0.0", server_port=2002, share=True, favicon_path='../logo.png')