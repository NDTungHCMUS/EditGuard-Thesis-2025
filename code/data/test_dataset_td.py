import os
import os.path as osp
import torch
import torch.utils.data as data
import data.util as util

import random
import numpy as np
from PIL import Image

# ----- VN Start -----
import global_variables
# ----- VN End -----

class imageTestDataset(data.Dataset):

    def __init__(self, opt):
        super(imageTestDataset, self).__init__()
        self.opt = opt
        self.half_N_frames = opt['N_frames'] // 2
        self.data_path = opt['data_path']
        self.bit_path = opt['bit_path']
        self.txt_path = self.opt['txt_path']
        self.num_image = self.opt['num_images']

        # ----- VN Start -----
        self.data_split_path = opt['split_path_ori']
        self.num_child_images = self.opt['num_child_images']
        # ----- VN End -----
        
        with open(self.txt_path) as f:
            self.list_image = f.readlines()
        self.list_image = [line.strip('\n') for line in self.list_image]
        l = len(self.list_image) // (self.num_child_images + 1)
        self.image_list_gt = self.list_image
        self.image_list_bit = self.list_image


    def __getitem__(self, index):
        # ----- VN Start -----
        ## Explaination: Calculate index for the child images
        parent_index = index // self.num_child_images
        index = index % self.num_child_images
        parent_path_GT = osp.join(self.data_split_path, self.image_list_gt[parent_index])
        # ----- ORIGINAL -----
        # path_GT = self.image_list_gt[index]
        # ----- VN End -----

        img_GT = util.read_img(None, osp.join(parent_path_GT, f"{index}.png"))
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float().unsqueeze(0)

        # ----- VN Start -----
        ## Explaination: Resize Image
        width = global_variables.TEST_CONFIG['datasets']['TD']['width']
        height = global_variables.TEST_CONFIG['datasets']['TD']['height']
        img_GT = torch.nn.functional.interpolate(img_GT, size=(width, height), mode='nearest', align_corners=None)
        ## ----- ORIGINAL -----
        # img_GT = torch.nn.functional.interpolate(img_GT, size=(512, 512), mode='nearest', align_corners=None)
        # ----- VN End -----

        T, C, W, H = img_GT.shape
        list_h = []
        R = 0
        G = 0
        B = 255
        image = Image.new('RGB', (W, H), (R, G, B))
        result = np.array(image) / 255.
        expanded_matrix = np.expand_dims(result, axis=0) 
        expanded_matrix = np.repeat(expanded_matrix, T, axis=0)
        imgs_LQ = torch.from_numpy(np.ascontiguousarray(expanded_matrix)).float()
        imgs_LQ = imgs_LQ.permute(0, 3, 1, 2)


        imgs_LQ = torch.nn.functional.interpolate(imgs_LQ, size=(W, H), mode='nearest', align_corners=None)

        list_h.append(imgs_LQ)

        list_h = torch.stack(list_h, dim=0)

        return {
                'LQ': list_h,
                'GT': img_GT
            }
    
    def __len__(self):
        # ----- VN Start -----
        return len(self.image_list_gt) * self.num_child_images
        # ----- ORIGINAL -----
        # return len(self.image_list_gt)
        # ----- VN End -----
