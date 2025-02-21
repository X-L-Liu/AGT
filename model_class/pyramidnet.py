import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    out_channel_ratio = 1

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)
            feature_size = shortcut.size()[2:4]
        else:
            shortcut = x
            feature_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.zeros(batch_size, residual_channel - shortcut_channel, feature_size[0], feature_size[1])
            out += torch.cat((shortcut, padding.to(shortcut.device)), 1)
        else:
            out += shortcut

        return out


class Bottleneck(nn.Module):
    out_channel_ratio = 4

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, (planes * 1), kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d((planes * 1))
        self.conv3 = nn.Conv2d((planes * 1), planes * Bottleneck.out_channel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * Bottleneck.out_channel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.bn4(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)
            feature_size = shortcut.size()[2:4]
        else:
            shortcut = x
            feature_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.zeros(batch_size, residual_channel - shortcut_channel, feature_size[0], feature_size[1])
            out += torch.cat((shortcut, padding.to(shortcut.device)), 1)
        else:
            out += shortcut

        return out


class PyramidNet(nn.Module):

    def __init__(self, dataset, depth, alpha, num_classes, bottleneck=False):
        super(PyramidNet, self).__init__()
        self.dataset = dataset
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            self.in_planes = 16
            if bottleneck:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock

            self.add_rate = alpha / (3 * n * 1.0)

            self.input_feature_dim = self.in_planes
            self.conv1 = nn.Conv2d(3, self.input_feature_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.input_feature_dim)

            self.feature_dim = self.input_feature_dim
            self.layer1 = self.pyramidal_make_layer(block, n)
            self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
            self.layer3 = self.pyramidal_make_layer(block, n, stride=2)

            self.final_feature_dim = self.input_feature_dim
            self.bn_final = nn.BatchNorm2d(self.final_feature_dim)
            self.relu_final = nn.ReLU(inplace=True)
            self.avg_pool = nn.AvgPool2d(8)
            self.fc = nn.Linear(self.final_feature_dim, num_classes)

        elif self.dataset == 'imagenet':
            blocks = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
            layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3],
                      200: [3, 24, 36, 3]}

            if layers.get(depth) is None:
                if bottleneck:
                    blocks[depth] = Bottleneck
                    temp_cfg = int((depth - 2) / 12)
                else:
                    blocks[depth] = BasicBlock
                    temp_cfg = int((depth - 2) / 8)

                layers[depth] = [temp_cfg, temp_cfg, temp_cfg, temp_cfg]
                print('=> the layer configuration for each stage is set to', layers[depth])

            self.in_planes = 64
            self.add_rate = alpha / (sum(layers[depth]) * 1.0)

            self.input_feature_dim = self.in_planes
            self.conv1 = nn.Conv2d(3, self.input_feature_dim, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.input_feature_dim)
            self.relu = nn.ReLU(inplace=True)
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.feature_dim = self.input_feature_dim
            self.layer1 = self.pyramidal_make_layer(blocks[depth], layers[depth][0])
            self.layer2 = self.pyramidal_make_layer(blocks[depth], layers[depth][1], stride=2)
            self.layer3 = self.pyramidal_make_layer(blocks[depth], layers[depth][2], stride=2)
            self.layer4 = self.pyramidal_make_layer(blocks[depth], layers[depth][3], stride=2)

            self.final_feature_dim = self.input_feature_dim
            self.bn_final = nn.BatchNorm2d(self.final_feature_dim)
            self.relu_final = nn.ReLU(inplace=True)
            self.fc = nn.Linear(self.final_feature_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        down_sample = None
        if stride != 1:
            down_sample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)

        layers = []
        self.feature_dim = self.feature_dim + self.add_rate
        layers.append(block(self.input_feature_dim, int(round(self.feature_dim)), stride, down_sample))
        for i in range(1, block_depth):
            temp_feature_dim = self.feature_dim + self.add_rate
            layers.append(
                block(int(round(self.feature_dim)) * block.out_channel_ratio, int(round(temp_feature_dim)), 1))
            self.feature_dim = temp_feature_dim
        self.input_feature_dim = int(round(self.feature_dim)) * block.out_channel_ratio

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            x = self.conv1(x)
            x = self.bn1(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.bn_final(x)
            x = self.relu_final(x)
            x = F.adaptive_avg_pool2d(x, output_size=1)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        elif self.dataset == 'imagenet':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.max_pool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.bn_final(x)
            x = self.relu_final(x)
            x = F.adaptive_avg_pool2d(x, output_size=1)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x


def PyramidNet110(num_classes):
    assert num_classes == 10, 'The num_classes of PyramidNet110 should be 10 according to the previous experience.'
    return PyramidNet(dataset='cifar10', depth=110, alpha=64, num_classes=num_classes, bottleneck=False)


def PyramidNet164(num_classes):
    assert num_classes == 100, 'The num_classes of PyramidNet164 should be 100 according to the previous experience.'
    return PyramidNet(dataset='cifar100', depth=164, alpha=48, num_classes=num_classes, bottleneck=True)


def PyramidNet200(num_classes):
    return PyramidNet(dataset='imagenet', depth=200, alpha=300, num_classes=num_classes, bottleneck=True)
