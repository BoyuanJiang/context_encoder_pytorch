from __future__ import print_function, division
import os
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from torch.utils.data import Dataset
# from mypath import Path
from torchvision import transforms
# from dataloaders import custom_transforms as tr
import random

class MICCAISegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 11
    # this is magic
    CLASSES = [0, 149, 178, 188, 108, 53, 149, 163, 240, 65, 128]
    MAPPING = dict(zip(CLASSES, range(NUM_CLASSES)))

    def __init__(self,
                 args,
                 base_dir='.',
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super(MICCAISegmentation, self).__init__()
        self._base_dir = base_dir

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = base_dir
        
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '_image.txt')), "r") as f:
                image_paths = f.read().splitlines()

            with open(os.path.join(os.path.join(_splits_dir, splt + '_label.txt')), "r") as f:
                label_paths = f.read().splitlines()

            self.images += image_paths
            self.categories += label_paths

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        # sample = {'image': _img, 'label': _target}
        sample = _img


        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)
        return sample


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _temp = np.array(Image.open(self.categories[index]).convert('L'))

        _target = Image.fromarray(self.encode_segmap(_temp))
        return _img, _target
    
    def encode_segmap(self, grey):
        # Put all void classes to zero
        mask = np.zeros(grey.shape, dtype=np.uint8)
        for _class in self.CLASSES:
            mask[grey == _class] = self.MAPPING[_class]
        return mask

    def transform_tr(self, sample):
        # composed_transforms = transforms.Compose([
        #     tr.RandomHorizontalFlip(),
        #     tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
        #     tr.RandomGaussianBlur(),
        #     tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #     tr.ToTensor()])
        composed_transforms = transforms.Compose([
                            RandomHorizontalFlip(),
                            RandomScaleCrop(base_size=self.args.imageSize, crop_size=self.args.crop_size),
                            # transforms.Resize(self.args.imageSize),
                            # transforms.CenterCrop(self.args.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                            ])
        # dataset = dset.ImageFolder(root=opt.dataroot, transform=transform )

        return composed_transforms(sample)
    
    def RandomHorizontalFlip(self, sample):
        img = sample
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        return img

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.ToTensor(), 
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        return composed_transforms(sample)

    def __str__(self):
        return 'MICCAI'

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img

# if __name__ == '__main__':
#     from dataloaders.utils import decode_segmap
#     from torch.utils.data import DataLoader
#     import matplotlib.pyplot as plt
#     import argparse

#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     args.base_size = 513
#     args.crop_size = 513

#     voc_train = VOCSegmentation(args, split='train')

#     dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

#     for ii, sample in enumerate(dataloader):
#         for jj in range(sample["image"].size()[0]):
#             img = sample['image'].numpy()
#             gt = sample['label'].numpy()
#             tmp = np.array(gt[jj]).astype(np.uint8)
#             segmap = decode_segmap(tmp, dataset='pascal')
#             img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
#             img_tmp *= (0.229, 0.224, 0.225)
#             img_tmp += (0.485, 0.456, 0.406)
#             img_tmp *= 255.0
#             img_tmp = img_tmp.astype(np.uint8)
#             plt.figure()
#             plt.title('display')
#             plt.subplot(211)
#             plt.imshow(img_tmp)
#             plt.subplot(212)
#             plt.imshow(segmap)

#         if ii == 1:
#             break

#     plt.show(block=True)


