import numpy as np
import argparse
import torch
import futility
import fmodel
import json

parser = argparse.ArgumentParser(description = "Predict model")
parser.add_argument('--input', default='./flowers/test/28/image_05253.jpg')
parser.add_argument('--checkpoint', default='./chckpt.pth')
parser.add_argument('--top_k', default=5)
parser.add_argument('--category_names', default='cat_to_name.json')
parser.add_argument('--gpu', default='gpu')


args = parser.parse_args()
filepath= args.input
checkpoint= args.checkpoint
top_k= args.top_k
category_names= args.category_names
processor= args.gpu


def main():
    model = fmodel.loadchkpt(checkpoint)

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    probs, classes = fmodel.predict(filepath,model,cat_to_name,processor)  
    
    #print(probs, classes)
    
    print("Predictions:")
    for i in range(args.top_k):
          print("#{: <3} {: <25} Prob: {:.2f}%".format(i, classes[i], probs[i]*100))
        
if __name__ == '__main__':
    main()