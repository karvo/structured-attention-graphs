import os
import torch
import shutil
import torch.nn as nn
from torchvision import models
from torchvision import datasets
import torch.nn.functional as F

from custom_model_architectures import custom_model_architecture

def load_model(model_path,model_name):

    model = None
    dataset_path = r'/media/kv/Documents/git/mtkvcs-dataset/datscan/'

    print(f"Model name: {model_name}\n")
    train_data = datasets.ImageFolder(os.path.join(dataset_path,'train'))
    class_names = train_data.classes
    print(f"Class names {class_names}\n")

    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)

        model.fc = nn.Linear(2048,2, bias=True)

        model.load_state_dict(torch.load(model_path))

    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        
        model.classifier = nn.Sequential(nn.Linear(25088,4096), # Configure the classifier
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096,4096, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096,len(class_names), bias=True))
    
    elif model_name == 'custom':
        model = custom_model_architecture()
        pass
    
    model.load_state_dict(torch.load(model_path))
    
    return model

def purge(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
