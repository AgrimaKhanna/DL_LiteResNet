import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F 
import numpy as np 
import torch.optim as optim
import torch.backends.cudnn as cudnn  


# Training
def train(net, trainloader, epoch, config, optimizer, criterion, writer, device='cpu'):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_losses = [] 
    train_acc = [] 
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if config["grad_clip"]: nn.utils.clip_grad_value_(net.parameters(), clip_value=config["grad_clip"]) 
        optimizer.step()

        train_loss += loss.item()
        train_losses.append(train_loss)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item() 

        train_acc.append(100.*correct/total) 
        print('Batch_idx: %d | Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'% (batch_idx, train_loss/(batch_idx+1), 100.*correct/total, correct, total)) 
    writer.add_scalar('Loss/train_loss', np.mean(train_losses), epoch) 
    writer.add_scalar('Accuracy/train_accuracy', np.mean(train_acc), epoch) 

    return net, optimizer, np.mean(train_losses), np.mean(train_acc)


# Testing 
def test(net, testloader, epoch, config, optimizer, criterion, writer,savename, best_acc, device='cpu'):
    # global best_acc
    net.eval()
    test_loss = 0
    test_losses = [] 
    test_acc = [] 
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            test_losses.append(test_loss)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item() 
            test_acc.append(100.*correct/total) 
            print('Batch_idx: %d | Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'% ( batch_idx, test_loss/(batch_idx+1), 100.*correct/total, correct, total)) 
        writer.add_scalar('Loss/test_loss', np.mean(test_losses), epoch) 
        writer.add_scalar('Accuracy/test_accuracy', np.mean(test_acc), epoch) 

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc: 
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'config': config
        }
        torch.save(state, os.path.join('../summaries/', savename, 'ckpt.pth'))
        best_acc = acc
    return net, optimizer, best_acc, np.mean(test_losses), np.mean(test_acc) 

# RESNET MODEL IS DEFINED HERE
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, conv_kernel_size=3, shortcut_kernel_size=1, drop=0.4):
        super(BasicBlock, self).__init__()
        self.drop = drop 
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=conv_kernel_size, stride=stride, padding=int(conv_kernel_size/2), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=conv_kernel_size,stride=1, padding=int(conv_kernel_size/2), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,kernel_size=shortcut_kernel_size, stride=stride, padding=int(shortcut_kernel_size/2), bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        if self.drop: self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        if self.drop: out = self.dropout(out)
        return out



def conv1x1(in_channels,
            out_channels,
            stride=1,
            groups=1,
            bias=False):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias)

class SEBlock(nn.Module):
    def __init__(self,
                 channels,
                 reduction=16):
        super(SEBlock, self).__init__()
        mid_cannels = channels // reduction

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = conv1x1(
            in_channels=channels,
            out_channels=mid_cannels,
            bias=True)
        self.activ = nn.ReLU(inplace=True) 

        self.conv2 = conv1x1(
            in_channels=mid_cannels,
            out_channels=channels,
            bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w
        return x


class ResNet(nn.Module):
    def __init__(
            self, 
            block, 
            num_blocks, 
            conv_kernel_sizes=None, 
            shortcut_kernel_sizes=None,
            num_classes=10, 
            num_channels=32, 
            avg_pool_kernel_size=4, 
            drop=None, 
            squeeze_and_excitation=None):
        super(ResNet, self).__init__()
        self.in_planes = num_channels
        # self.avg_pool_kernel_size = avg_pool_kernel_size 
        self.avg_pool_kernel_size = int(32 / (2**(len(num_blocks)-1)))
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(3, self.num_channels, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_channels) 

        self.drop = drop 
        self.squeeze_and_excitation = squeeze_and_excitation 

        if self.squeeze_and_excitation: 
            self.seblock = SEBlock(channels=self.num_channels) 
        self.residual_layers = [] 
        for n in range(len(num_blocks)): 
            stride = 1 if n==0 else 2 # stride=1 for first residual layer, and stride=2 for the remaining layers 
            conv_kernel_size = conv_kernel_sizes[n] if conv_kernel_sizes else 3 # setting default kernel size of block's convolutional layers 
            shortcut_kernel_size = shortcut_kernel_sizes[n] if shortcut_kernel_sizes else 1 # setting default kernel size of block's skip connection (shortcut) layers 
            self.residual_layers.append(self._make_layer(
                                                    block, 
                                                    self.num_channels*(2**n), 
                                                    num_blocks[n], 
                                                    stride=stride, 
                                                    conv_kernel_size=conv_kernel_size, 
                                                    shortcut_kernel_size=shortcut_kernel_size)) 

        self.residual_layers = nn.ModuleList(self.residual_layers)
        self.linear = nn.Linear(self.num_channels*(2**n)*block.expansion, num_classes) 
        if self.drop: 
            self.dropout = nn.Dropout(self.drop) # Define proportion or neurons to dropout

    def _make_layer(self, block, planes, num_blocks, stride, conv_kernel_size, shortcut_kernel_size):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, conv_kernel_size, shortcut_kernel_size, drop=self.drop))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.squeeze_and_excitation: out = self.seblock(out) 
        for layer in self.residual_layers: 
            out = layer(out)         
        out = F.avg_pool2d(out, self.avg_pool_kernel_size)
        out = out.view(out.size(0), -1)
        if self.drop: out = self.dropout(out)
        out = self.linear(out)
        return out


def ResNetLite(config=None): 
    # Best Model 
    net =  ResNet(
            block=BasicBlock, 
            num_blocks=[4, 4, 3],                          # N: number of Residual Layers | Bi:Residual blocks in Residual Layer i 
            conv_kernel_sizes=[3, 3, 3],            # Fi: Conv. kernel size in Residual Layer i 
            shortcut_kernel_sizes=[1, 1, 1] ,    # Ki: Skip connection kernel size in Residual Layer i 
            num_channels=64,                      # Ci: # channels in Residual Layer i 
            avg_pool_kernel_size=8,      # P: Average pool kernel size 
            drop=0,                                      # use dropout with drop proportion 
            squeeze_and_excitation=1   # Enable/disable Squeeze-and-Excitation Block 
        ) 
    
    total_params = 0 
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters())))) 
    return net, total_params 


 

def get_optimizer(net, hyperparam_optimizer, lr, momentum, weight_decay):
        # Setting the optimizer
    if hyperparam_optimizer == 'adam': 
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif hyperparam_optimizer == 'sgd': 
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay) 

    return optimizer

def get_scheduler(hyperparam_scheduler, optimizer):

    # Setting the scheduler
    if hyperparam_scheduler == 'CosineAnnealingLR': 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) 
    elif hyperparam_scheduler == 'LambdaLR': 
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.65 ** epoch)
    elif hyperparam_scheduler == 'MultiplicativeLR': 
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.65 ** epoch)
    elif hyperparam_scheduler == 'StepLR': 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1) 
    elif hyperparam_scheduler == 'MultiStepLR': 
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6,8,9], gamma=0.1) 
    elif hyperparam_scheduler == 'ExponentialLR': 
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1) 
    elif hyperparam_scheduler == 'CyclicLR': 
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular") 
    elif hyperparam_scheduler == 'CyclicLR2': 
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular2") 
    elif hyperparam_scheduler == 'CyclicLR3': 
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="exp_range",gamma=0.85) 
    elif hyperparam_scheduler == 'OneCycleLR': 
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10) 
    elif hyperparam_scheduler == 'OneCycleLR2': 
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10,anneal_strategy='linear') 
    elif hyperparam_scheduler == 'CosineAnnealingWarmRestarts': 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1) 
    
    return scheduler
