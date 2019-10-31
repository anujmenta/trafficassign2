from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--numworkers', type=int, default=4)
args = parser.parse_args()

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import * # data.py in the same folder
#initialize_data(args.data) # extracts the zip files, makes a validation set

if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
    use_gpu = False
    print("Using CPU")

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset(
    [
    datasets.ImageFolder(args.data + '/train_images', transform=data_transforms),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_rotate1),
    # datasets.ImageFolder(args.data + '/train_images', transform=data_transform_rotate2),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_colorjitter_brightness),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_colorjitter_saturation),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_colorjitter_contrast),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_colorjitter_hue),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_grayscale),
    # datasets.ImageFolder(args.data + '/train_images', transform=data_transform_pad),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_centercrop),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_shear),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_hrflip),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_vrflip),
    # datasets.ImageFolder(args.data + '/train_images', transform=data_transform_bothflip),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_translate),
    # datasets.ImageFolder(args.data + '/train_images', transform=data_transform_colorjitter_brightness_hflip),
    # datasets.ImageFolder(args.data + '/train_images', transform=data_transform_colorjitter_saturation_hflip),
    # datasets.ImageFolder(args.data + '/train_images', transform=data_transform_colorjitter_contrast_hflip),
    # datasets.ImageFolder(args.data + '/train_images', transform=data_transform_colorjitter_brightness_vflip),
    # datasets.ImageFolder(args.data + '/train_images', transform=data_transform_colorjitter_saturation_vflip),
    # datasets.ImageFolder(args.data + '/train_images', transform=data_transform_colorjitter_contrast_vflip),
    # datasets.ImageFolder(args.data + '/train_images', transform=data_transform_randomperspective),
    # datasets.ImageFolder(args.data + '/train_images', transform=data_transform_vflip_rotation),
    # datasets.ImageFolder(args.data + '/train_images', transform=data_transform_hflip_rotation),
    ]),batch_size=args.batch_size, shuffle=True, num_workers=args.numworkers, pin_memory=use_gpu)


val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=use_gpu)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net
model = Net()

if use_gpu:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

train_loss_track = []
val_loss_track = []
lr_tracker = []
accuracy_track = []

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if use_gpu:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss_track.append(loss.item())
    plt.figure(10)
    plt.plot(train_loss_track)
    plt.savefig('train_loss.png')

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if use_gpu:
            data = data.cuda()
            target = target.cuda()
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    val_loss_track.append(validation_loss)
    scheduler.step(validation_loss)
    #plot validation loss
    plt.figure(20)
    plt.plot(val_loss_track)
    plt.savefig('val_loss.png')
    #plot accuracy
    accuracy_track.append(100. * correct / len(val_loader.dataset))
    plt.figure(30)
    plt.plot(accuracy_track)
    plt.savefig('accuracy_track.png')
    print(accuracy_track)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    for param_group in optimizer.param_groups:
        lr_tracker.append(param_group['lr'])
    plt.figure(40)
    plt.plot(lr_tracker)
    plt.savefig('lr_tracker.png')
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')