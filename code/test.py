import os
import math
import argparse
import random
import logging
import copy
from configs import config, args
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import numpy as np
from utils.image_handler import split_all_images, combine_images_from_folder, combine_torch_tensors_4d, split_torch_tensors_4d, save_tensor_images, write_extracted_messages
from utils.random_walk import random_walk_unique
from utils.preprocess import load_pairs_from_file, load_copyright, tensor_to_binary_string
from utils.mapping import create_list_data
from utils.reed_solomons import compute_parity, recover_original

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
    # Take opt from config 
    opt = copy.deepcopy(config)

    # distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        rank = -1
        print('Disabled distributed training.')
    else:
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # Step 1: Split images into 36 sub-images (comment if already done)
    # split_all_images(input_folder = opt['datasets']['TD']['data_path'], output_folder = opt['datasets']['TD']['split_path_ori'], grid_size = 2)

    # ## create train and val dataloader
    # dataset_ratio = 200  # enlarge the size of each epoch
    # for phase, dataset_opt in opt['datasets'].items():
    #     print("phase", phase)
    #     if phase == 'TD':
    #         val_set = create_dataset(dataset_opt)
    #         val_loader = create_dataloader(val_set, dataset_opt, opt, None)
    #     elif phase == 'val':
    #         val_set = create_dataset(dataset_opt)
    #         val_loader = create_dataloader(val_set, dataset_opt, opt, None)
    #     else:
    #         raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    # # for i, batch in enumerate(val_loader):
    # #     print(i)
    # #     print("SHAPE OF a batch:", batch['LQ'].shape)

    # # # # create model
    # model = create_model(opt)
    # model.load_test(args.ckpt)
      
    # # # Create Random Walk
    # # random_walk_squeuence = random_walk_unique()
    # # print("RANDOM WALK:", random_walk_squeuence)

    # # # Load copyright and metadata from files
    # # list_copyright_metadata = load_pairs_from_file(opt['datasets']['TD']['copyright_path'])
    # # mapping_data = create_list_data(random_walk_squeuence, list_copyright_metadata[0][0], list_copyright_metadata[0][1])
    # # print("MAPPING_DATA:", mapping_data)


    # n = 2
    # # Create copyright and corresponding parity
    # list_copyright = load_copyright(opt['datasets']['TD']['copyright_path'])
    # list_parity = []
    # for i in range(len(list_copyright)):
    #   list_parity.append(compute_parity(list_copyright[i]))
    # print("LIST COPYRIGHT:", list_copyright)
    # print("LIST PARITY:", list_parity)

    # cnt_cannot_solve = 0
    # for parent_image_id, val_data in enumerate(val_loader):
    #     # img_dir = os.path.join('results',opt['name'])
    #     # util.mkdir(img_dir)

    #     # Step 1: Embed data into images
    #     list_container = []
    #     list_ref_L = []
    #     list_real_H = []
    #     list_messageTensor = []
    #     for i in range(0, n * n):
    #         child_data = {
    #             'LQ': val_data['LQ'][i].unsqueeze(0),
    #             'GT': val_data['GT'][i].unsqueeze(0)
    #         }
    #         # print("Child Data of parent: {parent_image_id}, child: {i} is:", child_data)
    #         # print("Shape LQ of Child Data:", child_data['LQ'].shape)
    #         # print("Shape GT of Child Data:", child_data['GT'].shape)
    #         model.feed_data(child_data)
    #         list_ref_L.append(model.ref_L)
    #         list_real_H.append(model.real_H)
    #         if (i % 2 == 0):
    #           message = list_copyright[i//2]
    #         else:
    #           message = list_parity[i//2]
    #         I_container, messageTensor = model.embed(message)
    #         list_messageTensor.append(messageTensor)
    #         print("MESSAGE:",message)
    #         list_container.append(I_container)

    #     # Step 1.1: Save n^2 images to folder
    #     # save_tensor_images(list_container, parent_image_id, opt['datasets']['TD']['split_path_con'])

    #     # Step 2: Combine n^2 images into one (4 dimensions)
    #     parent_container = combine_torch_tensors_4d(list_container, num_images = n * n)
    #     # parent_container = torch.nn.functional.interpolate(parent_container, size=(512, 512), mode='nearest', align_corners=None)
    #     # print("Shape của parent container: ", parent_container.shape)
    #     # print("Giá trị của Parent container: ", parent_container)

    #     # Step 2.1: Save parent_container to folder
    #     # parent_container_img = util.tensor2img(parent_container.detach()[0].float().cpu())
    #     # save_img_path = os.path.join(opt['datasets']['TD']['merge_path'],f'{str(parent_image_id).zfill(4)}.png')
    #     # util.save_img(parent_container_img, save_img_path)

    #     # Step 3: Diffusion on parent_container
    #     parent_y_forw, parent_y = model.diffusion(image_id = parent_image_id, y_forw = parent_container)

    #     # Step 3.1: Save parent_y_forw to folder
    #     # parent_rec_img = util.tensor2img(parent_y_forw)
    #     # save_img_path = os.path.join(opt['datasets']['TD']['merge_path'],f'{str(parent_image_id).zfill(4)}_diffusion.png')
    #     # util.save_img(parent_rec_img, save_img_path)

    #     # Step 4: Split parent_rec into n^2 images
    #     list_container_rec = split_torch_tensors_4d(parent_y_forw, grid_size = n)
    #     list_container_rec_quantize = split_torch_tensors_4d(parent_y, grid_size = n)
        
    #     # Step 4.1: Save n^2 images to folder
    #     save_tensor_images(list_container_rec, parent_image_id, opt['datasets']['TD']['split_path_rec'])
    #     # print("Shape of list_container_rec[0]:", list_container_rec[0].shape)
    #     # for i in range(len(list_container_rec)):
    #     #     print(f"List_container_rec {i}", list_container_rec[i])
            
    #     list_fake_H = []
    #     list_fake_H_h = []
    #     list_forw_L = []
    #     list_recmessage = []
    #     list_message = []
    #     print("LENGTH of list message: ", len(list_messageTensor))
    #     # Step 5: Extract from 36 images
    #     for i in range(0, n * n):
    #         fake_H, fake_H_h, forw_L, recmessage, message = model.extract(list_messageTensor[i], y_forw = list_container_rec[i], y = list_container_rec_quantize[i])
    #         list_fake_H.append(fake_H)
    #         list_fake_H_h.append(fake_H_h)
    #         list_forw_L.append(forw_L)
    #         list_recmessage.append(recmessage)
    #         list_message.append(message)

    #     # Step 5.1: Save all messages to file
    #     for i in range(0, n * n):
    #       list_message[i] = tensor_to_binary_string(list_message[i])
    #       list_recmessage[i] = tensor_to_binary_string(list_recmessage[i])
    #     write_extracted_messages(parent_image_id, list_message, list_recmessage, opt['datasets']['TD']['copyright_output'])

        
    #     # Step 6: Try to fix base on Reed-Solomons
        
    #     list_input_to_correct = []
    #     list_recmessage_fix = []
    #     # Build string to do reed-solomons
    #     for i in range(0, n * n, 2):
    #       list_input_to_correct.append(list_recmessage[i])
    #     for i in range(1, n * n, 2):
    #       list_input_to_correct[i//2] += list_recmessage[i]
    #     print("LIST SOLOMON:", list_input_to_correct)
    #     for i in range(0, n * n // 2):
    #       a = recover_original(str(list_input_to_correct[i]))
    #       print("DA CORRECT:", a)
    #       if (a == -1):
    #         cnt_cannot_solve += 1
    #       else:
    #         list_recmessage_fix.append(a[:64])
    #         list_recmessage_fix.append(a[64:])
    #     write_extracted_messages(parent_image_id, list_message, list_recmessage_fix, opt['datasets']['TD']['copyright_output_fix'])

    # print("CANNOT SOLVE:", cnt_cannot_solve)



        

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


if __name__ == '__main__':
    main()