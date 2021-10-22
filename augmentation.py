# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
from timm.data import create_transform

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

def get_augmentations(args):
    if args.aug == 'moco':
        return MocoAugmentations(args)
    if args.aug == 'barlow':
        return BarlowtwinsAugmentations(args)
    if args.aug == 'multicrop':
        return MultiCropAugmentation(args)
    if args.aug == 'multicropeval':
        return MultiCropEvalAugmentation(args)
    if args.aug == 'rand':
        return RandAugmentation(args)

class RandAugmentation(object):
    def __init__(self, args):
        self.aug = create_transform(
                input_size=224,
                is_training=True,
                color_jitter=0.4,
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation='bicubic',
                re_prob=0.25,
                re_mode='pixel',
                re_count=1,
            )

    def __call__(self, image):
        crops = []
        crops.append(self.aug(image))
        crops.append(self.aug(image))
        return crops

class MocoAugmentations(object):
    def __init__(self, args):
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.), interpolation=Image.BICUBIC),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]) 

    def __call__(self, image):
        crops = []
        crops.append(self.aug(image))
        crops.append(self.aug(image))
        return crops

class BarlowtwinsAugmentations(object):
    def __init__(self, args):
        self.aug1 = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.aug2 = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.aug1(image))
        crops.append(self.aug2(image))
        return crops

class MultiCropAugmentation(object):
    def __init__(self, args):
        global_crops_scale = args.global_crops_scale
        local_crops_scale  = args.local_crops_scale
        local_crops_number = args.local_crops_number

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            #utils.GaussianBlur(1.0),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            #utils.GaussianBlur(0.1),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            #utils.GaussianBlur(p=0.5),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

