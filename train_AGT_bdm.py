import argparse
import torch.optim as optim
import torchattacks
from torchvision import datasets
from random import randint
from model_class import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from unet import *


class Clean:
    def __init__(self):
        self.eps = 0

    def __call__(self, image, *args):
        return image


def train_epoch():
    def_model.train()
    top1 = AverageMeter()
    with tqdm(total=len(train_loader), desc='Train-Progress') as pbar:
        for k, (image, label) in enumerate(train_loader):
            image, label = torch.Tensor(image).to(device), torch.Tensor(label).to(device)
            for n in range(len(attacks)):
                attacks[n].eps = randint(4, 16) / 255
                image[n::len(attacks)] = attacks[n](image[n::len(attacks)], label[n::len(attacks)])
            logit = tsm_model(def_model(image))
            loss = F.cross_entropy(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1.update((logit.max(1)[1] == label).sum().item(), len(label))
            pbar.update(1)

    return top1.acc_rate


def test_epoch():
    def_model.eval()
    top1 = AverageMeter()
    with tqdm(total=len(test_loader), desc='Test--Progress') as pbar:
        for k, (image, label) in enumerate(test_loader):
            image, label = torch.Tensor(image).to(device), torch.Tensor(label).to(device)
            for n in range(len(attacks)):
                image[n::len(attacks)] = attacks[n](image[n::len(attacks)], label[n::len(attacks)])
            with torch.no_grad():
                logit = tsm_model(def_model(image))
            top1.update((logit.max(1)[1] == label).sum().item(), len(label))
            pbar.update(1)

    return top1.acc_rate


def main(Reload):
    global def_model_load_path
    for epoch in range(config.epochs):
        print(f'Reload: {Reload + 1}/{config.reload}  |  Epoch: {epoch + 1}/{config.epochs}')
        train_acc = train_epoch()
        test_acc = test_epoch()
        scheduler.step()
        if test_acc > config.best_acc:
            def_model_load_path = os.path.join(model_save_path, f'def_{config.AGT_num}_{test_acc:.4f}.pt')
            pre_path = def_model_load_path.replace(f'{test_acc:.4f}', f'{config.best_acc:.4f}')
            if os.path.exists(pre_path) and pre_path is not config.def_model_load_path:
                os.remove(pre_path)
            torch.save(def_model.state_dict(), def_model_load_path)
            config.best_acc = test_acc
        print(f'Train-Acc: {train_acc * 100:.2f}%  Test-Acc: {test_acc * 100:.2f}%  Best-Test-Acc: {config.best_acc * 100:.2f}%')
        print('-' * 85)


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'Dataset')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--milestones', type=tuple, default=[45, 75, 100], help='Stage of adjusting learning rate')
    parser.add_argument('--epochs', type=int, default=120, help='Total epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--reload', type=int, default=10, help='Reload model')
    parser.add_argument('--tar_model_name', type=str, default='ResNet34', help='The name of the classifier')
    parser.add_argument('--tar_model_load_path', type=str, default=r'')

    parser.add_argument('--def_model_load_path', type=str, default=r'')
    parser.add_argument('--tsm_load_path', type=str, default=r'')
    parser.add_argument('--def_model_save_path', type=str, default=r'DefenseModel')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'miniimagenet'])
    parser.add_argument('--AGT_num', type=int, default=1)

    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--best_acc', type=float, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    config = parse_args()
    device = torch.device(f'cuda:{config.device}')

    assert config.tar_model_load_path != ''
    assert config.tsm_load_path != ''
    if config.AGT_num > 1:
        assert config.def_model_load_path != ''

    if not os.path.exists(config.data_path):
        os.makedirs(config.data_path)
    if config.dataset_name == 'cifar10':
        num_classes = 10
        trainSet = datasets.CIFAR10(root=config.data_path, train=True, download=True, transform=trans_cifar10_train_64)
        testSet = datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=trans_cifar10_test_64)
        model_save_path = os.path.join(config.def_model_save_path, 'cifar10')
    else:
        num_classes = 100
        trainSet = MiniImageNet(root=config.data_path, train=True, pixel='96', transform=trans_miniimagenet_train_96)
        testSet = MiniImageNet(root=config.data_path, train=False, pixel='96', transform=trans_miniimagenet_test_96)
        model_save_path = os.path.join(config.def_model_save_path, 'miniimagenet')
    train_loader = DataLoader(trainSet, config.batch_size, True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(testSet, config.batch_size, False, pin_memory=True, num_workers=2)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    def_model_load_path = config.def_model_load_path

    tar_model = globals()[config.tar_model_name](num_classes)
    tar_model.load_state_dict(torch.load(config.tar_model_load_path, map_location=device))
    tar_model.to(device)
    tar_model.eval()
    attacks = [
        Clean(),
        torchattacks.FGSM(tar_model),
        torchattacks.PGD(tar_model, steps=10),
        torchattacks.DIFGSM(tar_model, steps=10)
    ]

    tsm_model = globals()[config.tar_model_name](num_classes)
    tsm_model.load_state_dict(torch.load(config.tsm_load_path, map_location=device))
    tsm_model.to(device)
    tsm_model.eval()

    for reload in range(config.reload):
        def_model = UNet_2Plus()
        if def_model_load_path != '':
            def_model.load_state_dict(torch.load(def_model_load_path, map_location=device))
        def_model.to(device)
        def_model.eval()
        print(f'Total params: {sum(p.numel() for p in def_model.parameters()) / 1000000.0:.2f}M')
        optimizer = optim.SGD(def_model.parameters(), config.lr, config.momentum, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.milestones)
        main(reload)
