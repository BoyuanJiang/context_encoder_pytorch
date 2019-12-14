import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset
import os
from torchvision import transforms


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['input'], sample['mask']

        # numpy: H x W x C
        # torch: C X H X W
        image, mask = image.transpose((2, 0, 1)), mask.transpose((2, 0, 1))
        return {'input': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}


class ImageDataset(Dataset):
    def __init__(self, input_dir, transformer=None):
        self.input_dir = input_dir
        self.transformer = transformer

    def __len__(self):
        return len([x for x in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, x))])

    # def __one_hot_encode(self, image):
    #     image = image // 32
    #     mask_ohe = np.zeros([image.shape[0], image.shape[1], 8])
    #     for c in range(8):
    #         mask_ohe[:, :, c] = (image == c).astype(np.float)
    #     return mask_ohe

    def __getitem__(self, idx):
        inputfn = self.input_dir + "/frame{0:03d}_input.png".format(idx)
        print(inputfn)
        print(self.input_dir)
        input_image = Image.open(inputfn).convert('RGB')
       #  input_image = transforms.ToPILImage()(input_image)
        # print(input_image)
        mask = Image.open(self.input_dir + "/frame{0:03d}_mask.png".format(idx)).convert('P')
        # mask_image = io.imread(mask_img_name)
        # mask = transforms.ToPILImage()(mask)      
        # mask = self.__one_hot_encode(mask)

        sample = {'input': input_image, 'mask': mask}

        if self.transformer:
            sample = self.transformer(sample)

        return sample


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().add(1).div(2).mul(255).clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = torch.div(batch, 255.0)
    batch -= Variable(mean)
    # batch /= Variable(std)
    batch = torch.div(batch,Variable(std))
    return batch
