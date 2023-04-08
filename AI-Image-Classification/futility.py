import torch
import json
from torchvision import datasets, transforms

def load_data(filepath):
    data_dir = filepath
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'

    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                         transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                              std = [0.229, 0.224, 0.225])])
    
    test_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                               std = [0.229, 0.224, 0.225])])
    
    valid_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                               std = [0.229, 0.224, 0.225])])    
    
    image_dataset_train = datasets.ImageFolder(train_dir, transform = train_transform)
    image_dataset_test = datasets.ImageFolder(test_dir, transform = test_transform)
    image_dataset_valid = datasets.ImageFolder(valid_dir, transform = valid_transform)
    
    
    dataloader_train = torch.utils.data.DataLoader(image_dataset_train, batch_size = 32, shuffle = True)
    dataloader_test = torch.utils.data.DataLoader(image_dataset_test, batch_size = 32, shuffle = True)
    dataloader_valid = torch.utils.data.DataLoader(image_dataset_valid, batch_size = 32)
    
    return dataloader_train, dataloader_test, dataloader_valid, image_dataset_train
