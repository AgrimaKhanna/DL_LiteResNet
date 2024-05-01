hyperparams = {
    "LiteResNet": {
        "conv_kernel_sizes": [3, 3, 3],
        "num_blocks": [4, 4, 3],
        "num_channels": 64,
        "shortcut_kernel_sizes": [1, 1, 1],
        "avg_pool_kernel_size": 8,
        "drop": 0,  # proportion for dropout
        "squeeze_and_excitation": 1,  # True=1, False=0
        "max_epochs": 200,
        "optimizer": "sgd",
        "lr_sched": "CosineAnnealingLR",
        "momentum": 0.9,
        "lr": 0.1,
        "weight_decay": 0.0005,
        "batch_size": 128,
        "num_workers": 14,
        "resume_ckpt": 0,  # 0 if not resuming, else path to checkpoint
        "data_augmentation": 1,  # True=1, False=0
        "data_normalize": 1,  # True=1, False=0
        "grad_clip": 0.1,
        "lookahead": 1
    },
    "resnet18": {
        "conv_kernel_sizes": [3, 3, 3, 3],
        "num_blocks": [2, 2, 2, 2],
        "num_channels": 64,
        "shortcut_kernel_sizes": [1, 1, 1, 1],
        "avg_pool_kernel_size": 4,
        "drop": 0,  # proportion for dropout
        "squeeze_and_excitation": 0,  # True=1, False=0
        "max_epochs": 200,
        "optimizer": "sgd",
        "lr_sched": "CosineAnnealingLR",
        "momentum": 0.9,
        "lr": 0.1,
        "weight_decay": 0.0005,
        "batch_size": 128,
        "num_workers": 14,
        "resume_ckpt": 0,  # 0 if not resuming, else path to checkpoint
        "data_augmentation": 0,  # True=1, False=0
        "data_normalize": 0,  # True=1, False=0
        "grad_clip": 0
    }
}
