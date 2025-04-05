import os
import math
import argparse
import random
import logging
import copy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import numpy as np

# ----- VN Start -----
## Explaination: Import library
from utils.random_walk import random_walk_unique
from utils.my_util import load_copyright_metadata_from_files, tensor_to_binary_string, compute_parity_from_list_copyright_metadata, compute_message, get_copyright_metadata_from_list_with_correction, get_copyright_metadata_from_list_without_correction, split_all_images, combine_torch_tensors_4d, split_torch_tensors_4d, write_extracted_messages
from utils.reed_solomons import compute_parity, recover_original
# ------ VN End ------

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def cal_pnsr(sr_img, gt_img):
    # calculate PSNR
    gt_img = gt_img / 255.
    sr_img = sr_img / 255.

    psnr = util.calculate_psnr(sr_img * 255, gt_img * 255)

    return psnr

def get_min_avg_and_indices(nums):
    # Get the indices of the smallest 1000 elements
    indices = sorted(range(len(nums)), key=lambda i: nums[i])[:900]
    
    # Calculate the average of these elements
    avg = sum(nums[i] for i in indices) / 900
    
    # Write the indices to a txt file
    with open("indices.txt", "w") as file:
        for index in indices:
            file.write(str(index) + "\n")
    
    return avg


def main():
    # Options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--ckpt', type=str, default='/userhome/NewIBSN/EditGuard_open/checkpoints/clean.pth', help='Path to pre-trained model.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # Distributed training settings
    if args.launcher == 'none':  # Disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    # Loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # Convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # ----- VN Start -----
    ## Explaination: Split images into n^2 sub-images (comment if already done)
    split_all_images(input_folder = opt['datasets']['TD']['data_path'],
                     output_folder = opt['datasets']['TD']['split_path_ori'],
                     num_child_images = opt['datasets']['TD']['num_child_images'],
                     num_images = opt['datasets']['TD']['num_images'])
    # ----- VN End -----

    # Create train and val dataloader
    dataset_ratio = 200  # Enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'TD':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

    # Create model
    model = create_model(opt)
    model.load_test(args.ckpt)
      
    # # # Create Random Walk
    # # random_walk_squeuence = random_walk_unique()
    # # print("RANDOM WALK:", random_walk_squeuence)

    # ----- VN Start -----
    ## Explaination: Clear output files after run again
    with open(opt['datasets']['TD']['copyright_output_without_correction'], 'w') as f:
        pass
    with open(opt['datasets']['TD']['copyright_output_with_correction'], 'w') as f:
        pass
    # ----- VN End -----

    # ----- VN Start -----
    ## Explaination: Create copyright, metadata and corresponding parity
    list_dict_copyright_metadata = load_copyright_metadata_from_files(opt['datasets']['TD']['copyright_path'])
    list_dict_parity_copyright_metadata = compute_parity_from_list_copyright_metadata(list_dict_copyright_metadata)
    # ----- VN End -----
    
    # ----- VN Start -----
    ## Explaination: Initialize neccessary variables
    cnt_cannot_solve_all = 0
    num_images = opt['datasets']['TD']['num_images']
    num_child_images = opt['datasets']['TD']['num_child_images']
    bit_error_list_without_correction_code = []
    bit_error_list_with_correction_code = []

    ## Explaination: Main flow
    for parent_image_id, val_data in enumerate(val_loader):  
        # Step 1: Embed data into images
        list_container = []
        list_messageTensor = []
        for i in range(0, num_child_images):
            child_data = {
                'LQ': val_data['LQ'][i].unsqueeze(0),
                'GT': val_data['GT'][i].unsqueeze(0)
            }
            model.feed_data(child_data)
            message = compute_message(i, list_dict_copyright_metadata[parent_image_id], list_dict_parity_copyright_metadata[parent_image_id])
            I_container, messageTensor = model.embed(message)
            list_messageTensor.append(messageTensor)
            list_container.append(I_container)

        # Step 1.1: Save n^2 images to folder
        for i in range(len(list_container)):
            child_container_img = util.tensor2img(list_container[i].detach()[0].float().cpu())
            folder_name = str(parent_image_id + 1).zfill(4)
            output_folder = os.path.join(opt['datasets']['TD']['split_path_con'], folder_name)
            save_img_path = os.path.join(output_folder,f'{str(i).zfill(4)}.png')
            util.save_img(child_container_img, save_img_path)

        # Step 2: Combine n^2 images into one (4 dimensions)
        parent_container = combine_torch_tensors_4d(list_container, num_images = num_child_images)

        # Step 2.1: Save parent_container to folder
        parent_container_img = util.tensor2img(parent_container.detach()[0].float().cpu())
        save_img_path = os.path.join(opt['datasets']['TD']['merge_path'],f'{str(parent_image_id + 1).zfill(4)}.png')
        util.save_img(parent_container_img, save_img_path)

        # Step 3: Diffusion on parent_container
        parent_y_forw, parent_y = model.diffusion(image_id = parent_image_id, y_forw = parent_container)

        # Step 3.1: Save parent_y_forw to folder
        parent_rec_img = util.tensor2img(parent_y_forw)
        save_img_path = os.path.join(opt['datasets']['TD']['merge_path'],f'{str(parent_image_id + 1).zfill(4)}_diffusion.png')
        util.save_img(parent_rec_img, save_img_path)

        # Step 4: Split parent_rec into n^2 images
        list_rec = split_torch_tensors_4d(parent_y_forw, num_child_images = num_child_images)
        list_rec_quantize = split_torch_tensors_4d(parent_y, num_child_images = num_child_images)
        
        # Step 4.1: Save n^2 images to folder
        for i in range(len(list_rec)):
            child_rec_img = util.tensor2img(list_rec[i].detach()[0].float().cpu())
            folder_name = str(parent_image_id + 1).zfill(4)
            output_folder = os.path.join(opt['datasets']['TD']['split_path_rec'], folder_name)
            save_img_path = os.path.join(output_folder,f'{str(i).zfill(4)}.png')
            util.save_img(child_rec_img, save_img_path)
            
        list_recmessage = []
        list_message = []
        # Step 5: Extract from n^2 child images
        for i in range(0, num_child_images):
            recmessage, message = model.extract(list_messageTensor[i], y_forw = list_rec[i], y = list_rec_quantize[i])
            list_recmessage.append(recmessage)
            list_message.append(message)

        # Step 5.1: Convert list_message, list_recmessage from tensor to binary string
        for i in range(0, num_child_images):
          list_message[i] = tensor_to_binary_string(list_message[i])
          list_recmessage[i] = tensor_to_binary_string(list_recmessage[i])
        
        # Step 5.2: Get copyright (before, after), metadata (before, after) from list_message, list_recmessage
        copyright_before, copyright_after, metadata_before, metadata_after = get_copyright_metadata_from_list_without_correction(list_message, list_recmessage)
        bit_error = write_extracted_messages(parent_image_id, copyright_before, copyright_after, metadata_before, metadata_after, opt['datasets']['TD']['copyright_output_without_correction'])
        bit_error_list_without_correction_code.append(bit_error)

        # Step 6: Try to fix base on Reed-Solomons        
        copyright_before, copyright_after, metadata_before, metadata_after, cnt_cannot_solve = get_copyright_metadata_from_list_with_correction(list_message, list_recmessage)
        bit_error = write_extracted_messages(parent_image_id, copyright_before, copyright_after, metadata_before, metadata_after, opt['datasets']['TD']['copyright_output_with_correction'])
        bit_error_list_with_correction_code.append(bit_error)
        cnt_cannot_solve_all += cnt_cannot_solve

    avg_bit_error_without_correction = sum(bit_error_list_without_correction_code) / len(bit_error_list_without_correction_code)
    avg_bit_error_with_correction = sum(bit_error_list_with_correction_code) / len(bit_error_list_with_correction_code)
    print(f"Cannot Solve {cnt_cannot_solve_all} pairs among {num_images * num_child_images // 2} pairs")
    print(f"FINAL RESULT:\n BIT_ERR WITHOUT CORRECTION IS: {avg_bit_error_without_correction} \n BIT_ERR WITH REED-SOLOMON CORRECTION IS: {avg_bit_error_with_correction}")

    # ----- ORIGINAL -----
    # img_dir = os.path.join('results',opt['name'])
        # util.mkdir(img_dir)
    # # validation
    # # avg_psnr = 0.0
    # # avg_psnr_h = [0.0]*opt['num_image']
    # # avg_psnr_lr = 0.0
    # # biterr = []
    # # idx = 0
    # for parent_image_id, val_data in enumerate(val_loader):
    #     img_dir = os.path.join('results',opt['name'])
    #     util.mkdir(img_dir)
    #     for i in range(0, 36):
    #         child_data = {
    #             'LQ': val_data['LQ'][i],
    #             'GT': val_data['GT'][i]
    #         }
    #         model.feed_data(child_data)
    #         model.test(image_id)

    #         visuals = model.get_current_visuals()

    #         t_step = visuals['SR'].shape[0]
    #         idx += t_step
    #         n = len(visuals['SR_h'])

    #         a = visuals['recmessage'][0]
    #         b = visuals['message'][0]

    #         # bitrecord = util.decoded_message_error_rate_batch(a, b)
    #         # print(bitrecord)
    #         # biterr.append(bitrecord)

    #         for i in range(t_step):

    #             sr_img = util.tensor2img(visuals['SR'][i])  # uint8
    #             sr_img_h = []
    #             for j in range(n):
    #                 sr_img_h.append(util.tensor2img(visuals['SR_h'][j][i]))  # uint8
    #             gt_img = util.tensor2img(visuals['GT'][i])  # uint8
    #             lr_img = util.tensor2img(visuals['LR'][i])
    #             lrgt_img = []
    #             for j in range(n):
    #                 lrgt_img.append(util.tensor2img(visuals['LR_ref'][j][i]))

    #             # Save SR images for reference
    #             save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(image_id, i, 'SR'))
    #             util.save_img(sr_img, save_img_path)

    #             for j in range(n):
    #                 save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:d}_{:s}.png'.format(image_id, i, j, 'SR_h'))
    #                 util.save_img(sr_img_h[j], save_img_path)

    #             save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(image_id, i, 'GT'))
    #             util.save_img(gt_img, save_img_path)

    #             save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(image_id, i, 'LR'))
    #             util.save_img(lr_img, save_img_path)

    #             for j in range(n):
    #                 save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:d}_{:s}.png'.format(image_id, i, j, 'LRGT'))
    #                 util.save_img(lrgt_img[j], save_img_path)

    #             # psnr = cal_pnsr(sr_img, gt_img)
    #             # psnr_h = []
    #             # for j in range(n):
    #             #     psnr_h.append(cal_pnsr(sr_img_h[j], lrgt_img[j]))
    #             # psnr_lr = cal_pnsr(lr_img, gt_img)

    #             # avg_psnr += psnr
    #             # for j in range(n):
    #             #     avg_psnr_h[j] += psnr_h[j]
    #             # avg_psnr_lr += psnr_lr

    # # avg_psnr = avg_psnr / idx
    # # avg_biterr = sum(biterr) / len(biterr)
    # # print(get_min_avg_and_indices(biterr))

    # # avg_psnr_h = [psnr / idx for psnr in avg_psnr_h]
    # # avg_psnr_lr = avg_psnr_lr / idx
    # # res_psnr_h = ''
    # # for p in avg_psnr_h:
    # #     res_psnr_h+=('_{:.4e}'.format(p))
    # # print('# Validation # PSNR_Cover: {:.4e}, PSNR_Secret: {:s}, PSNR_Stego: {:.4e},  Bit_Error: {:.4e}'.format(avg_psnr, res_psnr_h, avg_psnr_lr, avg_biterr))

    # ----- VN End -----


if __name__ == '__main__':
    main()