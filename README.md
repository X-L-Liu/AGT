# Alternating Guided Training for Robust Adversarial Defense

[//]: # ([Paper]&#40;&#41; )

> **Abstract:** *Adversarial examples can lead neural networks to produce high-confidence incorrect predictions
> in image classification tasks. To defend against adversarial example attacks, we propose a deep learning defense
> method known as alternating guided training (AGT). AGT employs a fully convolutional neural network as the basic
> defense model (BDM) and utilizes a pre-trained classification model (e.g., ResNet) as the target substitute
> model (TSM). Through a series of iterative processes, the BDM and TSM are alternately trained to enhance the
> perturbation elimination capability of the former and to correct the classification decision boundary of the
> latter, thereby achieving an overall robust adversarial defense. In black-box scenarios utilizing ResNet-34
> as the classification model, AGT achieves average defense rates exceeding 95.04% and 73.98% on CIFAR-10 and
> Mini-ImageNet, respectively, exhibiting the SOTA performance. *
>

## Installation

```
conda create -n AGT python=3.11.5
conda activate AGT
conda install --yes --file requirements.txt
```
Our project is not version-sensitive. It can typically run on other versions of above packages as well. 
Furthermore, allow the system to automatically select the version when installing any other missing libraries.

## Pre-Training Classifiers

You can train various classifiers with `train_pretrained_model.py`.

* For ResNet34 on CIFAR-10
  `python train_pretrained_model.py --classifier_name ResNet34 --dataset_name cifar10`
* For GoogLeNet on CIFAR-10
  `python train_pretrained_model.py --classifier_name GoogLeNet --dataset_name cifar10`

## The TSM guides the training of the BDM

* For ResNet34 on CIFAR-10
  `python train_AGT_bdm.py --tar_model_name XXXX --tar_model_load_path XXXX --def_model_load_path XXXX --tsm_load_path XXXX  --dataset_name  cifar10  --AGT_num 1`

## The BDM guides the training of the TSM

* For ResNet34 on CIFAR-10
  `python train_AGT_tsm.py --tar_model_name XXXX --tar_model_load_path XXXX --def_model_load_path XXXX --tsm_load_path XXXX  --dataset_name  cifar10  --AGT_num 1`
