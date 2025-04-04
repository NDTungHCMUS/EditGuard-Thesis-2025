'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data

def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        # ----- VN Start -----
        return torch.utils.data.DataLoader(dataset, batch_size=dataset_opt['num_child_images'], shuffle=False, num_workers=1,
                                           pin_memory=True)
        # ----- ORIGINAL -----
        # return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
        #                                    pin_memory=True)
        # ----- VN End -----


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'test':
        from data.coco_test_dataset import imageTestDataset as D
    elif mode == 'train':
        from data.coco_dataset import CoCoDataset as D
    elif mode == 'td':
        from data.test_dataset_td import imageTestDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    print(mode)
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
