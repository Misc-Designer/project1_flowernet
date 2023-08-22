import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

#Classifier for the model
class Classifier(nn.Module):
    
    def __init__(self, hidden, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, 102)
        self.dropout = nn.Dropout(p=0.25)
        
    def forward(self, inputs):
        inputs = inputs.view(inputs.shape[0], -1)
        
        inputs = self.dropout(F.relu(self.fc1(inputs)))
        outputs = F.log_softmax(self.fc2(inputs), dim=1)
        
        return outputs
    

def build_model(args):
    device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"
    #check for checkpoint directory, if present attempt to load the checkpoint provided
    chkpt = ''
    if(args.chkpt_loc != None):
        chkpt = torch.load(args.chkpt_loc)
        args.hidden = chkpt['hidden_layers']
        args.model_name = chkpt['model_name']

    #determine which model will be used based on user feedback
    model = None
    in_features = None
    if(args.model_name == 'vgg16'):
        model = models.vgg16(pretrained = True)
        in_features = int(model.classifier[0].in_features)
    elif(args.model_name == 'densenet161'):
        model = models.densenet161(pretrained = True)
        in_features = int(model.classifier.in_features)
    elif(args.model_name == 'efficientnet_b0'):
        model = models.efficientnet_b0(pretrained = True)
        in_features = int(model.classifier[1].in_features)

    if(model == None):
        raise ValueError('ERROR: did not type one of three model types currently implemented in training algorithm.')
    
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = Classifier(args.hidden, in_features)
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=args.learn_rate, momentum=args.momentum)
    loss_criteria = nn.NLLLoss()

    model.to(device)
    #finish load checkpoint, if present
    if(args.chkpt_loc != None):
        model.load_state_dict(chkpt['model_state_dict'])
        optimizer.load_state_dict(chkpt['optimizer_state_dict'])
        
        #credit to https://github.com/pytorch/pytorch/issues/2830 from github user "dogancan"
        #loads the optimizer to the gpu properly
        if(args.gpu == True):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
                        
    return model, loss_criteria, optimizer

#loads model for predictive inference
def build_predict_model(args):
    device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"
    
    chkpt = torch.load(args.checkpoint)
    args.hidden = chkpt['hidden_layers']
    args.model_name = chkpt['model_name']

    #determine which model will be used based on checkpoint data
    model = None
    in_features = None
    if(args.model_name == 'vgg16'):
        model = models.vgg16(pretrained = True)
        in_features = int(model.classifier[0].in_features)
    elif(args.model_name == 'densenet161'):
        model = models.densenet161(pretrained = True)
        in_features = int(model.classifier.in_features)
    elif(args.model_name == 'efficientnet_b0'):
        model = models.efficientnet_b0(pretrained = True)
        in_features = int(model.classifier[1].in_features)

    if(model == None):
        raise ValueError('ERROR: checkpoint is not from one of three model types currently implemented in training algorithm.')
    
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = Classifier(args.hidden, in_features)
    model.load_state_dict(chkpt['model_state_dict'])
    model.to(device)
    
    return model, chkpt['class_to_idx']
