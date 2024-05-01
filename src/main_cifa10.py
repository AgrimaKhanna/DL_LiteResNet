# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn  
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import yaml
import math
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
# from ResNetLite import ResNetLite 
from lookahead import Lookahead 
import config

if __name__ == '__main__': 
    # Setting the arguments
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model', default='LiteResNet', type=str, help='name of resnet architecture from config') 
    args = parser.parse_args()

    hyperparams = config.hyperparams[args.model]
    # Setting the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Setting the the best accuracy to 0 and start epoch to 0
    best_acc = 0  # best test accuracy
    start_epoch = 0 # start from epoch 0
    # Setting the experiment name
    exp = args.model
    # Preparing the dataset
    train_transformations = [transforms.ToTensor()]
    test_transformations = [transforms.ToTensor()]
    # Adding the transforms based on the hyperparameters
    if hyperparams["data_augmentation"]: 
        train_transformations.append(transforms.RandomCrop(32, padding=4)) 
        train_transformations.append(transforms.RandomHorizontalFlip()) 
    if hyperparams["data_normalize"]: 
        train_transformations.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))) 
        test_transformations.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))) 
    # Adding the transforms to the dataset - will be used for both train and test
    train_transformations = transforms.Compose(train_transformations) 
    test_transformations = transforms.Compose(test_transformations) 
    # Getting the train set and loading it
    train_dataset = torchvision.datasets.CIFAR10(root='../data_10', train=True, download=True, transform=train_transformations)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True, num_workers=hyperparams["num_workers"])
    # Getting the test set and loading it
    test_dataset = torchvision.datasets.CIFAR10(root='../data_10', train=False, download=True, transform=test_transformations)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(hyperparams["batch_size"]/4), shuffle=False, num_workers=hyperparams["num_workers"])
    # Creating the ResNet model and getting total number of parameters
    resnet_model, total_params = utils.ResNetLite(config=hyperparams) 
    hyperparams['total_params'] = total_params 
    print('The Total number of parameters in the model are: ', total_params) 
    # Moving the model to the device
    resnet_model = resnet_model.to(device)
    # Getting cuda if available
    if device == 'cuda':
        resnet_model = torch.nn.DataParallel(resnet_model)
        cudnn.benchmark = True
    
    if ("weights_init_type" in hyperparams): 
        def init_weights(m, type='default'): 
            if (isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)) and hasattr(m, 'weight'): 
                if type == 'xavier_uniform_': torch.nn.init.xavier_uniform_(m.weight.data)
                elif type == 'normal_': torch.nn.init.normal_(m.weight.data, mean=0, std=0.02)
                elif type == 'xavier_normal': torch.nn.init.xavier_normal(m.weight.data, gain=math.sqrt(2))
                elif type == 'kaiming_normal': torch.nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
                elif type == 'orthogonal': torch.nn.init.orthogonal(m.weight.data, gain=math.sqrt(2))
                elif type == 'default': pass 
        resnet_model.apply(lambda m: init_weights(m=m, type=hyperparams["weights_init_type"])) 

    if hyperparams["resume_ckpt"]:        
        # Load checkpoint.
        print('Getting the last checkpoint and training')
        checkpoint = torch.load(hyperparams["resume_ckpt"])
        resnet_model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    criterion = nn.CrossEntropyLoss()

    # Setting the optimizer
    optimizer = utils.get_optimizer(resnet_model, hyperparams['optimizer'], hyperparams['lr'], hyperparams['momentum'], hyperparams['weight_decay'])

    # Setting the lookahead optimizer
    if ("lookahead" in hyperparams) and hyperparams["lookahead"]: optimizer = Lookahead(optimizer, k=5, alpha=0.5) # Initialize Lookahead 

    # Setting the scheduler
    scheduler = utils.get_scheduler(hyperparams['lr_sched'],optimizer)

    # Setting the writer to write the results
    result_logs = SummaryWriter('../results/'+exp) 
 
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    # Training and testing for each epoch and saving the best model (based on test accuracy)
    for epoch in range(start_epoch, hyperparams["max_epochs"]): 
        # Training the model
        resnet_model, optimizer, train_loss, train_acc = utils.train(resnet_model, train_loader, epoch, hyperparams, optimizer, criterion, result_logs, device)
        # Testing the model
        resnet_model, optimizer, best_acc, test_loss, test_acc = utils.test(resnet_model, test_loader, epoch, hyperparams, optimizer, criterion, result_logs, exp, best_acc ,device)
        scheduler.step()

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
    
    # Saving to a text file
    with open(f"../results/results_{exp}_cifar10.txt", "w") as file:
        file.write(str(history))
    result_logs.close()  