import os
import pickle
from datetime import datetime

import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset

from .autoaugment import *
from .cutout import Cutout
from torch.utils.data import Dataset


trans_cifar10_train_32 = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=12)
])


trans_cifar10_test_32 = transforms.Compose([
    transforms.ToTensor()
])

trans_cifar10_train_64 = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=12),
    transforms.Resize((64, 64), antialias=True)
])


trans_cifar10_test_64 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64), antialias=True)
])


trans_miniimagenet_train_96 = transforms.Compose([
    transforms.RandomCrop(96, padding=12, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=20),
])

trans_miniimagenet_test_96 = transforms.Compose([
    transforms.ToTensor()
])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.acc_num = 0
        self.total_num = 0
        self.acc_rate = 0

    def update(self, acc_num, total_num):
        self.acc_num += acc_num
        self.total_num += total_num
        self.acc_rate = self.acc_num / self.total_num


class SelfPrint:
    def __init__(self, print_name=None):
        if print_name is None:
            print_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log_file = f'logs/{print_name}.txt'

    def __call__(self, info):
        print(info)
        with open(self.log_file, 'a') as file:
            print(info, file=file)


class MiniImageNet(VisionDataset):
    def __init__(self, root, train=True, pixel='64', transform=None):
        super().__init__(root, transform=transform)
        data = pickle.load(file=open(os.path.join(self.root, 'mini-imagenet-' + pixel + '.pkl'), 'rb'))
        if train:
            self.sample, self.label = data['train_sample'], data['train_label']
        else:
            self.sample, self.label = data['test_sample'], data['test_label']
        self.sample = self.sample.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index: int):
        img, target = self.sample[index], self.label[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        return len(self.label)


class AEDataset(Dataset):
    def __init__(self, file_paths: list, train=True, test_num=None):

        for file_path in file_paths:
            assert os.path.exists(file_path)

        self.train = train
        self.test_num = test_num

        AE = torch.load(file_paths[0])
        self.cle_img, self.adv_img = AE['cle_example'].unsqueeze(1), AE['adv_example'].unsqueeze(1)
        self.labels = AE['labels'].unsqueeze(1)

        if len(file_paths) > 1:
            for file_path in file_paths[1:]:
                AE = torch.load(file_path)
                self.cle_img = torch.cat((self.cle_img, AE['cle_example'].unsqueeze(1)), dim=1)
                self.adv_img = torch.cat((self.adv_img, AE['adv_example'].unsqueeze(1)), dim=1)
                self.labels = torch.cat((self.labels, AE['labels'].unsqueeze(1)), dim=1)

    def __len__(self):
        return len(self.cle_img)

    def __getitem__(self, item):
        if self.train:
            r = random.randint(0, self.adv_img.shape[1]-1)
        else:
            r = self.test_num
        return self.cle_img[item, r, ...] / 255, self.adv_img[item, r, ...] / 255, self.labels[item, r]
