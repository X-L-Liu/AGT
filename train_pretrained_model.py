import argparse
import time
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import *
from model_class import *


def train_epoch():
    model.train()
    top1 = AverageMeter()
    with tqdm(total=len(train_loader), desc='Train-Progress', ncols=100) as pbar:
        for k, (image, label) in enumerate(train_loader):
            image, label = torch.Tensor(image).to(device), torch.Tensor(label).to(device)
            if config.use_amp:
                with torch.cuda.amp.autocast():
                    logit = model(image)
                    loss = F.cross_entropy(logit, label)
                loss = scaler.scale(loss)
            else:
                logit = model(image)
                loss = F.cross_entropy(logit, label)
            optimizer.zero_grad()
            loss.backward()
            if config.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            top1.update((logit.max(1)[1] == label).sum().item(), len(label))
            pbar.update(1)

    return top1.acc_rate


def test_epoch():
    model.eval()
    top1 = AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Test-Progress ', ncols=100) as pbar:
            for k, (image, label) in enumerate(test_loader):
                image, label = torch.Tensor(image).to(device), torch.Tensor(label).to(device)
                logit = model(image)
                top1.update((logit.max(1)[1] == label).sum().item(), len(label))
                pbar.update(1)

    return top1.acc_rate


def main(Reload):
    global model_load_path
    for epoch in range(config.epochs):
        start = time.time()
        train_acc = train_epoch()
        test_acc = test_epoch()
        scheduler.step()
        if test_acc > config.best_acc:
            model_load_path = os.path.join(classifier_save_path, f'{config.classifier_name}_{test_acc:.4f}.pt')
            torch.save(model.state_dict(), model_load_path)
            if os.path.exists(model_load_path.replace(f'{test_acc:.4f}', f'{config.best_acc:.4f}')):
                os.remove(model_load_path.replace(f'{test_acc:.4f}', f'{config.best_acc:.4f}'))
            config.best_acc = test_acc
        print(f'Model: {config.classifier_name}  '
              f'Reload: {Reload + 1}/{config.reload}  Epoch: {epoch + 1}/{config.epochs}  '
              f'Train-Top1: {train_acc * 100:.2f}%  Test-Top1: {test_acc * 100:.2f}%  '
              f'Best-Top1: {config.best_acc * 100:.2f}%  Time: {time.time() - start:.0f}s')


def load_model():
    classifier = globals()[config.classifier_name](num_classes)
    if model_load_path != '':
        classifier.load_state_dict(torch.load(model_load_path, map_location=device))
    classifier.to(device)
    classifier.eval()

    return classifier


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'Dataset')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--milestones', type=tuple, default=(40, 70, 100))
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--reload', type=int, default=10)
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'miniimagenet'])
    parser.add_argument('--model_save_path', type=str, default=r'PretrainedModel')
    parser.add_argument('--model_load_path', type=str, default=r'')
    parser.add_argument('--best_acc', type=float, default=0)
    parser.add_argument('--classifier_name', type=str, default='ResNet34', choices=[])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_amp', type=bool, default=True)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    config = parse_args()
    device = torch.device(f'cuda:{config.device}')
    model_load_path = config.model_load_path

    if not os.path.exists(config.data_path):
        os.makedirs(config.data_path)
    if config.dataset_name == 'cifar10':
        num_classes = 10
        trainSet = datasets.CIFAR10(root=config.data_path, train=True, transform=trans_cifar10_train_64, download=True)
        testSet = datasets.CIFAR10(root=config.data_path, train=False, transform=trans_cifar10_test_64, download=True)
        classifier_save_path = os.path.join(config.model_save_path, 'cifar10')
    else:
        num_classes = 100
        trainSet = MiniImageNet(root=config.data_path, train=True, pixel='96', transform=trans_miniimagenet_train_96)
        testSet = MiniImageNet(root=config.data_path, train=False, pixel='96', transform=trans_miniimagenet_test_96)
        classifier_save_path = os.path.join(config.model_save_path, 'miniimagenet')
    train_loader = DataLoader(trainSet, config.batch_size, True, pin_memory=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(testSet, config.batch_size, False, pin_memory=True, num_workers=2, drop_last=True)

    if not os.path.exists(classifier_save_path):
        os.makedirs(classifier_save_path)

    for reload in range(config.reload):
        model = load_model()
        print('>' * 100)
        print(f'Model: {config.classifier_name}  '
              f'Total params: {sum(p.numel() for p in model.parameters()) / 1000000.0:.2f}M')
        optimizer = optim.SGD(model.parameters(), config.lr, config.momentum, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.milestones)
        scaler = torch.cuda.amp.GradScaler()
        main(reload)
