
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
from torch import optim, nn
from torchvision import datasets, transforms, models
import time
import argparse
import futility
import fmodel

parser = argparse.ArgumentParser(description = "Train model")
parser.add_argument('--data_dir', default='./flowers')
parser.add_argument('--save_dir', default='./chckpt.pth')
parser.add_argument('--arch', default='vgg16')
parser.add_argument('--learning_rate', default=0.001)
parser.add_argument('--hidden_units', default=512)
parser.add_argument('--dropout', default=0.5)
parser.add_argument('--epochs', default=3)
parser.add_argument('--gpu', default='gpu')


args = parser.parse_args()
filepath = args.data_dir
arch = args.arch
save_dir = args.save_dir
hidden_units = args.hidden_units
lr = args.learning_rate
dropout = args.dropout
processor = args.gpu

def main():
    
    dataloaders_train, dataloader_test, dataloaders_valid,image_datasets_train = futility.load_data(filepath)
    model, criterion = fmodel.load_model(arch, dropout, hidden_units, lr)
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
    #Train model
    epochs = 10
    print_every = 80
    steps =0
    cuda = torch.cuda.is_available()

    validation = True

    if cuda and processor == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    
    print('-----Training start------')
    start = time.time()

    for e in range (epochs):
        train_loss = 0
        steps=0            
        model.train()
    
    for (images, labels) in dataloaders_train:
        steps+=1
        images,labels = images.to(device), labels.to(device)
        model = model.to(device)
        
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step() 
        train_loss+= loss.item()
        
    #print("Epoch: {}/{} .. ".format(e+1,epochs),
    #      "Loss: {:.3f} .. ".format(train_loss/len(dataloaders_train)))
        if steps % print_every == 0:
            accuracy = 0
            evalloss =0
            model.eval()

            with torch.no_grad():
                for (img,label) in dataloaders_valid:
                    img = img.to(device)
                    label = label.to(device)
                    logps = model(img)
                
                    evalloss+= criterion(logps, label).item()
                
                    output = torch.exp(logps)
                
                    top_p, top_class = output.topk(1,dim=1)
                    equals = top_class == label.view(*top_class.shape)
                
                    accuracy+= torch.mean(equals.type(torch.FloatTensor)).item()
                
                    model.train()

                    print(#"Epoch: {}/{} .. ".format(e+1,epochs),
                          #"Loss: {:.3f} .. ".format(train_loss/len(dataloaders_train)),
                          "Validation Loss: {:.3f} .. ".format(evalloss/len(dataloaders_valid)),
                          "Validation Accuracy:{:.3f}".format(accuracy/len(dataloaders_valid)))
        
        model.class_to_idx = image_datasets_train.class_to_idx
        model.cpu()
        fmodel.save_checkpoint(save_dir, model, arch,hidden_units, dropout, lr,epochs,optimizer)
        #total_time = time.time()- start
        #print('training time ={:.0f}m {:.0f}s'.format(total_time//60, total_time%60))

if __name__ == "__main__":
    main()