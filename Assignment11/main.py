from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


# CUDA?
cuda = torch.cuda.is_available()


# Train Phase transformations
train_transforms = A.Compose([
                      A.PadIfNeeded(min_height=36, min_width=36, always_apply=True),
                      A.RandomCrop(height=32, width=32, always_apply=True),
                      #A.HorizontalFlip(p=0.5),
                      #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                      A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=8, min_width=8
                                      , fill_value=(0.49139968, 0.48215841, 0.44653091),mask_fill_value = None),
                      #A.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1, p=0.2),
                      A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
                      ToTensorV2()
                              ])

# Test Phase transformations
test_transforms = A.Compose([
                    A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
                    ToTensorV2()
                          ])

class Albumentations_CIFAR10(datasets.CIFAR10):
  def __init__(self, root="../data", train=True, download=True, transform=None):
    super().__init__( root=root, train=train, download=download, transform=transform)

  def __getitem__(self, index):
    image, label = self.data[index], self.targets[index]

    if self.transform is not None:
      transformed = self.transform(image=image)
      image = transformed["image"]
    return image, label

from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, criterion, epoch):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))
  return train_losses, train_acc

def test(model, device, test_loader,  criterion ):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_losses, test_acc



