
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import copy
import random
import os
        
class DataLoad:
    def __init__(self, ROOT, SEED, BATCH_SIZE, VAL_RATIO):
        self.ROOT = ROOT
        if not os.path.exists(ROOT):
            os.mkdir(ROOT)
        
        self.SEED = SEED
        self.BATCH_SIZE = BATCH_SIZE
        self.VAL_RATIO = VAL_RATIO
        self.dataload = self.load_data(self.ROOT,self.SEED, self.BATCH_SIZE, self.VAL_RATIO)


    def set_seed(self, SEED):    
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

    def get_mean_std(self, ROOT):
        # Data Normalization
        # MNIST data load from local files
        temp_train_data = datasets.MNIST(root=ROOT,
                                    train=True,
                                    download=True,
                                    transform=transforms.ToTensor())

        MEAN = temp_train_data.data.float().mean() / 255
        STD = temp_train_data.data.float().std() / 255
        return MEAN, STD

    def load_data(self, ROOT, SEED, BATCH_SIZE, VAL_RATIO):
        self.set_seed(SEED)
        MEAN, STD = self.get_mean_std(ROOT)
            
        # Augmentation 정의
        train_transforms = transforms.Compose([
            transforms.RandomRotation(5, fill=(0,)),
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[MEAN], std=[STD])
        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[MEAN], std=[STD])
        ])

        ## Load DataSet
        train_data = datasets.MNIST(root=ROOT,
                                    train=True,
                                    download=True,
                                    transform=train_transforms)

        test_data = datasets.MNIST(root=ROOT,
                                train=False,
                                download=True,
                                transform=test_transforms)

        ## Split Train Validation


        n_train = int(len(train_data) * (1-VAL_RATIO))
        n_val = len(train_data) - n_train

        train_data, val_data = data.random_split(train_data, [n_train, n_val])

        val_data = copy.deepcopy(val_data)
        val_data.dataset.transform = test_transforms


        # Data Loaders

        train_iterator = data.DataLoader(train_data,
                                        shuffle=True,
                                        batch_size=BATCH_SIZE)

        valid_iterator = data.DataLoader(val_data,
                                        batch_size=BATCH_SIZE)

        test_iterator = data.DataLoader(test_data,
                                        batch_size=BATCH_SIZE)

        print(f"Train Data: {len(train_data)}")
        print(f"Validation Data: {len(val_data)}")
        print(f"Test Data: {len(test_data)}")
        
        return train_iterator, valid_iterator, test_iterator
