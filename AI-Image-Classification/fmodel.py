import torchvision.transforms as transforms
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
from torch import optim, nn
from torchvision import datasets, transforms, models
import time
import argparse
from PIL import Image


def load_model(arch='vgg16', dropout=0.1,hidden_units=4096, lr=0.001):
    
    if arch == 'vgg16':
        model=models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model=models.densenet121(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad= False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1',nn.Linear(25088,hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout1',nn.Dropout(dropout)),
        ('fc2',nn.Linear(hidden_units,102)),    
        ('out', nn.LogSoftmax(dim=1)) 
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
    return model, criterion
    
def save_checkpoint(filepath='./chckpt.pth', model = 0, arch = 'vgg16', hidden_units = 4096, dropout = 0.3, lr = 0.001, epochs = 1, optimizer= 0):
    #model.class_to_idx = image_datasets_train.class_to_idx
    #model.cpu()
    # TODO: Save the checkpoint 
    checkpoint = {
        'epochs':epochs,
        #'input_size':25088,
        #'output_size':102,
        'arch':arch,
        'class_to_idx':model.class_to_idx,
        'state_dict':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'classifier': model.classifier
    }
    torch.save(checkpoint, filepath)

def loadchkpt(filepath):
    chkpt = torch.load(filepath)
    if chkpt['arch'] == 'vgg16':
        model=models.vgg16(pretrained=True)
    elif chkpt['arch'] == 'densenet121':
        model=models.densenet121(pretrained=True)
    #model=models.vgg16(pretrained=True)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    for param in model.parameters():
        param.requires_grad= False
    
    #model = models.Network(chkpt[epoch],
    #                       chkpt[input_size], 
    #                       chkpt[output_size],
    #                       chkpt[class_to_idx])
    
    model.classifier = chkpt['classifier']
    model.load_state_dict(chkpt['state_dict'])
    model.class_to_idx = chkpt['class_to_idx']
    #optimizer.load_state_dict(chkpt['optimizer'])
    
    return model 

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''  
    trp_image = Image.open(image)
    img_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    trp_image = img_transform(trp_image)
    return trp_image


def predict(image_path, model, cat_name,device='gpu', top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
        
    if device == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    processed_image=process_image(image_path)
        
    #processed_image= torch.from_numpy(processed_image).type(torch.FloatTensor)
    model_input = processed_image.unsqueeze_(0)
    
    #forward pass
    ps = torch.exp(model(model_input))
    top_prob,top_class=ps.topk(top_k)
    
    
    idx_to_flower = {v:cat_name[k] for k, v in model.class_to_idx.items()}
    predicted_flowers_list = [idx_to_flower[i] for i in top_class.tolist()[0]]

    # returning both as lists instead of torch objects for simplicity
    return top_prob.tolist()[0], predicted_flowers_list
    
    
    #conver to list
    #top_prob = top_prob.detach().tolist()[0]
    #top_class = top_class.detach().tolist()[0]
    
    #convert labels of model class to indices
    #idx_to_class = {value: key for key, value in model.class_to_idx.items()}
     
    #top_labels =[]
    #top_flower =[]
    
    #retrive top labels
    #for i in top_class:
     #   top_labels.append(idx_to_class[i])
    
    #retrive names of flowers
    #for j in top_class:
    #    top_flower.append([idx_to_class[j]])

    #return top_prob, top_flower

