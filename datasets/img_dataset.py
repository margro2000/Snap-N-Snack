import torch
from torch.utils.data import Dataset
from unidecode import unidecode
import os
import pickle as pkl
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np


class FoodImgs(Dataset):

    def __init__(
            self,
            imgs_path="../data/snapnsnackdb/simple_images",
            target_dict="../data/snapnsnackdb/targets_dict.pkl",
    ):
        self.targets_dict = pkl.load(open(target_dict, "rb"))
        self.imgs_paths = self.get_imgs_path(imgs_path)
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, idx, with_vis=False):
        im_label, im_path = self.imgs_paths[idx]
        # img = plt.imread(im_path)
        img_pil = Image.open(im_path).convert("RGB")
        # if len(img.shape) < 3 or img.shape[2] < 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #     print(img.shape)
        # if len(img.shape) == 4:
        #     img = np.asarray(Image.open(im_path).convert("RGB"))
        #     print(img.shape)
        # img = img.asarray

        if with_vis:
            print(im_path)
            plt.imshow(img_pil)
            plt.show()
        # img = self.transform(Image.fromarray(img))
        img = self.transform(img_pil)

        target = self.targets_dict[im_label][1:]
        return img, target

    def inv_transform(self, img):
        topil = T.ToPILImage()
        unnorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return topil(unnorm(img))

    def __len__(self):
        return len(self.imgs_paths)


    def filt(self, string):
        string = unidecode(string)
        string = string.strip().replace(" ", "+").replace(",", "")
        string = ''.join(e for e in string if e.isalnum())
        return string

    def get_imgs_path(self, imgs_path):
        imgs_paths = []
        for path in os.listdir(imgs_path):
            for f in os.listdir(os.path.join(imgs_path, path)):
                if ".jpg" in f or ".jpeg" in f:
                    category = os.path.basename(path)
                    category = self.filt(category)
                    if not torch.any(torch.isnan(self.targets_dict[category])):
                        imgs_paths.append((category, os.path.join(imgs_path, path, f)))

        return imgs_paths

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor