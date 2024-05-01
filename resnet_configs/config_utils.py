import yaml, numpy as np, copy 
from pprint import pprint 


ResNet18_default_config = {
  "avg_pool_kernel_size": 4, 
  "conv_kernel_sizes": [3, 3, 3, 3],
  "num_blocks": [2, 2, 2, 2] ,
  "num_channels": 64,
  "shortcut_kernel_sizes": [1, 1, 1, 1] ,
  "drop": 0, 
  "squeeze_and_excitation": 0, 
  "max_epochs": 200,
  "optim": "sgd" ,
  "lr_sched": "CosineAnnealingLR",
  "momentum": 0.9,
  "lr": 0.1 ,
  "weight_decay": 0.0005 ,
  "batch_size": 128,
  "num_workers": 16,
  "resume_ckpt": 0,
  "data_augmentation": 1, 
  "data_normalize": 1, 
  "grad_clip": 0.1 
} 

Vanilla_default_config = {
  "avg_pool_kernel_size": 4, 
  "conv_kernel_sizes": [3, 3, 3, 3],
  "num_blocks": [2, 2, 2, 2] ,
  "num_channels": 64,
  "shortcut_kernel_sizes": [1, 1, 1, 1],
  "drop": 0, 
  "squeeze_and_excitation": 0, 
  "max_epochs": 200,
  "optim": "sgd" ,
  "lr_sched": "CosineAnnealingLR",
  "momentum": 0.9,
  "lr": 0.1 ,
  "weight_decay": 0.0005 ,
  "batch_size": 128,
  "num_workers": 16,
  "resume_ckpt": 0,
  "data_augmentation": 0, 
  "data_normalize": 0, 
  "grad_clip": 0 
} 

good_default_config = {
  "avg_pool_kernel_size": 4, 
  "conv_kernel_sizes": [3, 3, 3, 3],
  "num_blocks": [2, 2, 2, 2] ,
  "num_channels": 64,
  "shortcut_kernel_sizes": [1, 1, 1, 1],
  "drop": 0.4, 
  "squeeze_and_excitation": 0, 
  "max_epochs": 200,
  "optim": "sgd" ,
  "lr_sched": "CosineAnnealingLR",
  "momentum": 0.9,
  "lr": 0.1 ,
  "weight_decay": 0.0005 ,
  "batch_size": 128,
  "num_workers": 16,
  "resume_ckpt": 0,
  "data_augmentation": 1, 
  "data_normalize": 1, 
  "grad_clip": 0.1 
} 

se_and_drop_on_good_default_config = {
  "avg_pool_kernel_size": 4, 
  "conv_kernel_sizes": [3, 3, 3, 3],
  "num_blocks": [2, 2, 2, 2] ,
  "num_channels": 64,
  "shortcut_kernel_sizes": [1, 1, 1, 1],
  "drop": 0.4, 
  "squeeze_and_excitation": 0, 
  "max_epochs": 200,
  "optim": "sgd" ,
  "lr_sched": "CosineAnnealingLR",
  "momentum": 0.9,
  "lr": 0.1 ,
  "weight_decay": 0.0005 ,
  "batch_size": 128,
  "num_workers": 16,
  "resume_ckpt": 0,
  "data_augmentation": 1, 
  "data_normalize": 1, 
  "grad_clip": 0.1 
} 
config = {} 

for name in ["se_fulldrop_good_ResNet4"]: 
  for num_blocks in [[2,1,1,1], [1,1,1,1]]: 
    for squeeze_and_excitation in [0,1]: 
      for drop in [0, 0.2, 0.4, 0.6, 0.8]: 
          exp = name 
          exp += "_num_blocks" + ['x'.join(str(x) for x in num_blocks)][0]
          exp += "_squeeze_and_excitation" + str(squeeze_and_excitation) 
          exp += "_drop" + str(drop) 
          config[exp] = copy.deepcopy(good_default_config)
          config[exp]['num_blocks'] = copy.deepcopy(num_blocks) 
          config[exp]['squeeze_and_excitation'] = squeeze_and_excitation  
          config[exp]['drop'] = drop 
print(len(config.keys()))
pprint(config.keys())

with open('resnet_configs/se_fulldrop_good_ResNet4.yaml', 'w') as file:
    yaml.dump(config, file) 

with open('resnet_configs/sunday_vanilla_ResNets2.yaml', 'w') as file:
    yaml.dump(config, file) 



