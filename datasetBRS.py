import os
import cv2
import glob
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from utils import remap_label

from imgaug import augmenters as iaa
from augs import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,
)


def get_augmentation(rng):
    shape_augs = [
        # * order = ``0`` -> ``cv2.INTER_NEAREST``
        # * order = ``1`` -> ``cv2.INTER_LINEAR``
        # * order = ``2`` -> ``cv2.INTER_CUBIC``
        # * order = ``3`` -> ``cv2.INTER_CUBIC``
        # * order = ``4`` -> ``cv2.INTER_CUBIC``
        # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding
        iaa.Affine(
            # scale images to 80-120% of their size, individually per axis
#             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate by -A to +A percent (per axis)
            translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
            shear=(-5, 5),  # shear by -5 to +5 degrees
            rotate=(-179, 179),  # rotate by -179 to +179 degrees
            order=0,  # use nearest neighbour
            backend="cv2",  # opencv for fast processing\
            seed=rng,
        ),
        # set position to 'center' for center crop
        # else 'uniform' for random crop
        iaa.Fliplr(0.5, seed=rng),
        iaa.Flipud(0.5, seed=rng),
    ]
    
    trans_augs = [iaa.Affine(
                # scale images to 80-120% of their size, individually per axis
#                 scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # translate by -A to +A percent (per axis)
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                order=0,  # use nearest neighbour
                backend="cv2",  # opencv for fast processing\
                seed=rng,),
              ]

    input_augs = [
        iaa.OneOf(
            [
                iaa.Lambda(
                    seed=rng,
                    func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                ),
                iaa.Lambda(
                    seed=rng,
                    func_images=lambda *args: median_blur(*args, max_ksize=3),
                ),
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                ),
            ]
        ),
        iaa.Sequential(
            [
                iaa.Lambda(
                    seed=rng,
                    func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                ),
                iaa.Lambda(
                    seed=rng,
                    func_images=lambda *args: add_to_saturation(
                        *args, range=(-0.2, 0.2)
                    ),
                ),
                iaa.Lambda(
                    seed=rng,
                    func_images=lambda *args: add_to_brightness(
                        *args, range=(-26, 26)
                    ),
                ),
                iaa.Lambda(
                    seed=rng,
                    func_images=lambda *args: add_to_contrast(
                        *args, range=(0.75, 1.25)
                    ),
                ),
            ],
            random_order=True,
        ),
    ]

    return shape_augs, input_augs, trans_augs

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root_dataset, imgMaxSize):
        # parse options
        self.imgMaxSize = imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling

        # parse the input list
        self.list_sample = self.parse_input_list(root_dataset)

        # mean and std
        self.normalize = transforms.Normalize(
            mean = [0.689, 0.504, 0.653],
            std = [0.200, 0.228, 0.177]
        )

    def parse_input_list(self, root_dataset):
        list_sample = glob.glob('%s/*%s' % (root_dataset, '.npy'))

        self.num_sample = len(list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))
        return list_sample

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
#         img = torch.from_numpy(np.array(img))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long()
        return segm
    
    def cropping_center(self, x, crop_shape, batch=False):
        """Crop an input image at the centre.
        Args:
            x: input array
            crop_shape: dimensions of cropped array

        Returns:
            x: cropped array
        """
        orig_shape = x.shape
        if not batch:
            h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
            w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
            x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
        else:
            h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
            w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
            x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
        return x
    
class GlobalCDataset(BaseDataset):
    def __init__(self, root_dataset, cfg):
        super(GlobalCDataset, self).__init__(root_dataset, cfg.imgMaxSize)
        self.root_dataset = root_dataset
        # down sampling rate of segm labe
        self.segm_downsampling_rate = cfg.segm_downsampling_rate
        self.imgReSizes = cfg.imgReSizes

        # classify images into two classes: 1. h > w and 2. h <= w
        
        self.augmentor = get_augmentation(0)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
    
    def __augmentations(self, img):
        shape_augs = self.shape_augs.to_deterministic()
        input_augs = self.input_augs.to_deterministic()
        img = shape_augs.augment_image(img)
        img = input_augs.augment_image(img)
        img = self.cropping_center(img, [self.imgMaxSize, self.imgMaxSize])
        img = self.img_transform(img)
        return img

    def __getitem__(self, index):
        import cv2
        cv2.setNumThreads(0)

        # load image only
        record_path = self.list_sample[index]

        record = np.load(record_path)

        img = record[...,:3]

        # augmentations
        img = img.astype(np.uint8)
        img = self.cropping_center(img, [self.imgReSizes[0], self.imgReSizes[0]])
#         img_h = cv2.resize(img, (self.imgMaxSize*2, self.imgMaxSize*2), interpolation = cv2.INTER_NEAREST)
#         img_l = cv2.resize(img, (int(self.imgMaxSize/2), int(self.imgMaxSize/2)), interpolation = cv2.INTER_NEAREST)
        
        img1 = self.__augmentations(img)
        img2 = self.__augmentations(img)
        
        return [img1, img2]

    def __len__(self):
        return self.num_sample

class NucleiCDataset(BaseDataset):
    def __init__(self, root_dataset, cfg):
        super(NucleiCDataset, self).__init__(root_dataset, cfg.imgMaxSize)
        self.root_dataset = root_dataset
        # down sampling rate of segm labe
        self.segm_downsampling_rate = cfg.segm_downsampling_rate
        self.imgReSizes = cfg.imgReSizes
        self.mode = cfg.mode
        self.fixed_sp = cfg.fixed_sp
        self.learned_sp = cfg.learned_sp
        self.sp_dir = cfg.train_data_sp_dir

        # classify images into two classes: 1. h > w and 2. h <= w
        
        self.augmentor = get_augmentation(0)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.trans_augs = iaa.Sequential(self.augmentor[2])
        
        if self.fixed_sp:
            region_size = 32
            x_len = int(self.imgReSizes[0] / region_size)
            y_len = int(self.imgReSizes[0] / region_size)
            target = np.ones([self.imgReSizes[0] ,self.imgReSizes[0] ], dtype=np.int16)
            for i in range(x_len):
                for j in range(y_len):
                    target[i*region_size: (i+1)*region_size, j*region_size: (j+1)*region_size] = i*x_len + j
            self.sp_map = target
    
    def __augmentations(self, imgs, spms):
        shape_augs = self.shape_augs.to_deterministic()
        input_augs = self.input_augs.to_deterministic()
        trans_augs = self.trans_augs.to_deterministic()
        imgs_aug = []
        spms_aug = []
        for img in imgs:
            img = shape_augs.augment_image(img)
            img = input_augs.augment_image(img)
            if 'dc' in self.mode:
                img = trans_augs.augment_image(img)
            img = self.cropping_center(img, [self.imgMaxSize, self.imgMaxSize])
            img = self.img_transform(img)
            imgs_aug.append(img)
        for spm in spms:
            spm = shape_augs.augment_image(spm)
            if 'dc' in self.mode:
                spm = trans_augs.augment_image(spm)
            spm = self.cropping_center(spm, [self.imgMaxSize, self.imgMaxSize])
            spm = cv2.resize(spm, (int(self.imgMaxSize/self.segm_downsampling_rate), 
                                   int(self.imgMaxSize/self.segm_downsampling_rate)), 
                                   interpolation = cv2.INTER_NEAREST)
            spm = self.segm_transform(spm)
            spms_aug.append(spm)
        return imgs_aug, spms_aug

    def __getitem__(self, index):
        import cv2
        cv2.setNumThreads(0)

        # load image only
        record_path = self.list_sample[index]
        record = np.load(record_path)
        img = record[...,:3]
 
        if self.fixed_sp: # using fix_winow
            spm = self.sp_map
        elif self.learned_sp:
            sp_path = os.path.join(self.sp_dir, os.path.basename(record_path))
            spm = np.load(sp_path)
        else:
            spm = record[...,-1]

        # augmentations
        img = img.astype(np.uint8)
        
        imgs = [self.cropping_center(img, [self.imgReSizes[0], self.imgReSizes[0]])]
        spms = [self.cropping_center(spm, [self.imgReSizes[0], self.imgReSizes[0]])]
        for i in range(len(self.imgReSizes)):
            if self.imgReSizes[i] == img.shape[0]:
                continue
            img_resized = cv2.resize(img, (self.imgReSizes[i], self.imgReSizes[i]), interpolation = cv2.INTER_NEAREST)
            img_resized = self.cropping_center(img_resized, [self.imgReSizes[0], self.imgReSizes[0]])
            
            spm_resized = cv2.resize(spm, (self.imgReSizes[i], self.imgReSizes[i]), interpolation = cv2.INTER_NEAREST)
            spm_resized = self.cropping_center(spm_resized, [self.imgReSizes[0], self.imgReSizes[0]])
            
            imgs.append(img_resized)
            spms.append(spm_resized)
        
        imgs1, spms1 = self.__augmentations(imgs, spms)
        imgs2, spms2 = self.__augmentations(imgs, spms)
        
        return imgs1, imgs2, spms1, spms2

    def __len__(self):
        return self.num_sample
    

class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, cfg):
        super(TrainDataset, self).__init__(root_dataset, cfg.imgMaxSize)
        self.root_dataset = root_dataset
        # down sampling rate of segm labe
        self.segm_downsampling_rate = cfg.segm_downsampling_rate

        # classify images into two classes: 1. h > w and 2. h <= w
        
        self.augmentor = get_augmentation(0)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])

    def __getitem__(self, index):
        import cv2
        cv2.setNumThreads(0)
        
        self.segMaxSize = self.imgMaxSize // self.segm_downsampling_rate
        
        shape_augs = self.shape_augs.to_deterministic()
        input_augs = self.input_augs.to_deterministic()

        # load image and label
        record_path = self.list_sample[index]
#         sp_path = self.list_sample_sp[index]
#         assert(os.path.basename(record_path) == os.path.basename(sp_path))

        record = np.load(record_path)
#         sp_record = np.load(sp_path)

        img = record[...,:3]
        segm = record[...,-2]
#         spm = sp_record[...,0]
        assert(img.shape[0] == segm.shape[0])
        assert(img.shape[1] == segm.shape[1])

        # augmentations
        img = img.astype(np.uint8)
        img = shape_augs.augment_image(img)
        segm = shape_augs.augment_image(segm)
#         spm = shape_augs.augment_image(spm)

        img = input_augs.augment_image(img)

        img = self.cropping_center(img, [self.imgMaxSize, self.imgMaxSize])
        segm = self.cropping_center(segm, [self.imgMaxSize, self.imgMaxSize])
#         spm = self.cropping_center(spm, [self.imgMaxSize, self.imgMaxSize])
        
#         spm = remap_label(spm)
        # resize seg map to output size
        
        if self.segm_downsampling_rate != 1:
            segm = cv2.resize(segm, (self.segMaxSize, self.segMaxSize), interpolation = cv2.INTER_NEAREST)

        # image transform, to torch float tensor 3xHxW

        img = self.img_transform(img)

        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
#         spm = self.segm_transform(spm)
        
        return [img, segm]

    def __len__(self):
        return self.num_sample


class ValDataset(BaseDataset):
    def __init__(self, root_dataset, cfg):
        super(ValDataset, self).__init__(root_dataset, cfg.imgMaxSize)
        self.root_dataset = root_dataset

    def __getitem__(self, index):
        import cv2
        cv2.setNumThreads(0)
        
        # load image and label
        record_path = self.list_sample[index]
#         sp_path = self.list_sample_sp[index]
#         assert(os.path.basename(record_path) == os.path.basename(sp_path))
        
        record = np.load(record_path)
#         sp_record = np.load(sp_path)
        
        img = record[...,:3]
        segm = record[...,-2]
#         spm = sp_record[...,0]
        
        assert(img.shape[0] == segm.shape[0])
        assert(img.shape[1] == segm.shape[1])

        # segm transform, to torch long tensor HxW
        
        img = self.cropping_center(img, [self.imgMaxSize, self.imgMaxSize])
        segm = self.cropping_center(segm, [self.imgMaxSize, self.imgMaxSize])
#         spm = self.cropping_center(spm, [self.imgMaxSize, self.imgMaxSize])
#         spm = remap_label(spm)
        
        img = self.img_transform(img)
        segm = self.segm_transform(segm)
#         spm = self.segm_transform(spm)

        return [img, segm]

    def __len__(self):
        return self.num_sample

class TrainDataset1(BaseDataset):
    def __init__(self, root_dataset, cfg):
        super(TrainDataset1, self).__init__(root_dataset, cfg.imgMaxSize)
        self.root_dataset = root_dataset
        # down sampling rate of segm labe
        self.segm_downsampling_rate = cfg.segm_downsampling_rate

        # classify images into two classes: 1. h > w and 2. h <= w
        
        self.augmentor = get_augmentation(0)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])

    def __getitem__(self, index):
        import cv2
        cv2.setNumThreads(0)
        
        self.segMaxSize = self.imgMaxSize // self.segm_downsampling_rate
        
        shape_augs = self.shape_augs.to_deterministic()
        input_augs = self.input_augs.to_deterministic()

        # load image and label
        record_path = self.list_sample[index]
#         sp_path = self.list_sample_sp[index]
#         assert(os.path.basename(record_path) == os.path.basename(sp_path))

        record = np.load(record_path)
#         sp_record = np.load(sp_path)

        img = record[...,:3]
        segm = record[...,-1]
#         spm = sp_record[...,0]
        assert(img.shape[0] == segm.shape[0])
        assert(img.shape[1] == segm.shape[1])

        # augmentations
        img = img.astype(np.uint8)
        img = shape_augs.augment_image(img)
        segm = shape_augs.augment_image(segm)
#         spm = shape_augs.augment_image(spm)

        img = input_augs.augment_image(img)

        img = self.cropping_center(img, [self.imgMaxSize, self.imgMaxSize])
        segm = self.cropping_center(segm, [self.imgMaxSize, self.imgMaxSize])
#         spm = self.cropping_center(spm, [self.imgMaxSize, self.imgMaxSize])
        
#         spm = remap_label(spm)
        # resize seg map to output size
        
        if self.segm_downsampling_rate != 1:
            segm = cv2.resize(segm, (self.segMaxSize, self.segMaxSize), interpolation = cv2.INTER_NEAREST)

        # image transform, to torch float tensor 3xHxW

        img = self.img_transform(img)

        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
#         spm = self.segm_transform(spm)
        
        return [img, segm]

    def __len__(self):
        return self.num_sample


class ValDataset1(BaseDataset):
    def __init__(self, root_dataset, cfg):
        super(ValDataset1, self).__init__(root_dataset, cfg.imgMaxSize)
        self.root_dataset = root_dataset

    def __getitem__(self, index):
        import cv2
        cv2.setNumThreads(0)
        
        # load image and label
        record_path = self.list_sample[index]
#         sp_path = self.list_sample_sp[index]
#         assert(os.path.basename(record_path) == os.path.basename(sp_path))
        
        record = np.load(record_path)
#         sp_record = np.load(sp_path)
        
        img = record[...,:3]
        segm = record[...,-1]
#         spm = sp_record[...,0]
        
        assert(img.shape[0] == segm.shape[0])
        assert(img.shape[1] == segm.shape[1])

        # segm transform, to torch long tensor HxW
        
        img = self.cropping_center(img, [self.imgMaxSize, self.imgMaxSize])
        segm = self.cropping_center(segm, [self.imgMaxSize, self.imgMaxSize])
#         spm = self.cropping_center(spm, [self.imgMaxSize, self.imgMaxSize])
#         spm = remap_label(spm)
        
        img = self.img_transform(img)
        segm = self.segm_transform(segm)
#         spm = self.segm_transform(spm)

        return [img, segm]

    def __len__(self):
        return self.num_sample
