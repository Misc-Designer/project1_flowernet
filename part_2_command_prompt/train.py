#imports
from preprocess import create_datasets
from build_model import build_model
import argparse
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms, models

#set up argument parsing
parser = argparse.ArgumentParser(description='Train a three-layer Neural Network Classifier built onto a pre-existing \
image recognition model. NOTE: make sure you have backups of checkpoints in chkpts folder.')
parser.add_argument('data_dir', action="store", help='The location of the dataset, pre-sorted.')
parser.add_argument('--save_dir', action="store", dest="save_dir", default="chkpts/", help='The location to which \
the model will be saved.')
parser.add_argument('--checkpoint_location', action="store", dest="chkpt_loc", default = None, help='Location of a model checkpoint to be \
used for further training')
parser.add_argument('--arch', action="store", dest="model_name", default="vgg16", help='The model to be used as the \
base model for the classifier. Options: "vgg16", "densenet161", and "efficientnet_b0".')
parser.add_argument('--learning_rate', action="store", dest="learn_rate", type=float, default=0.003, help='The learning rate for the \
model. Default 0.003.')
parser.add_argument('--hidden_units', action="store", dest="hidden", default=1024, type=int, help='Total number of hidden units \
in the hidden layer between input and outputs. Default 1024.')
parser.add_argument('--epochs', action="store", dest="epoch", type=int, default=1, help='Number of training epochs the model \
should perform before finishing training and saving checkpoint. Default 1.')
parser.add_argument('--momentum', action="store", dest="momentum", type=float, default=0.9, help='This model uses Stochastic Gradient \
Descent as its optimizer. The momentum for this optimizer. Default 0.9.')
parser.add_argument('--gpu', action="store_true", dest="gpu", default=False, help='Use this argument if you have \
a GPU that can run the model and wish to use it.')

args = parser.parse_args()

#pull in datasets
trainloader, validloader, testloader, class_to_idx = create_datasets(args.data_dir)

#build the model
model, loss_criteria, optimizer = build_model(args)
device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"

#model variables for use later
steps = 0
running_loss = 0
print_by = 5
#train the model
for epoch in range(args.epoch):
    for inputs, labels in trainloader:
        steps+=1
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        train_loss = loss_criteria(output, labels)
        train_loss.backward()
        optimizer.step()
        
        running_loss += train_loss.item()
        
        if steps % print_by == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs_test, labels_test in validloader:
                    inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                    output = model.forward(inputs_test)
                    batch_loss = loss_criteria(output, labels_test)
                    prob = torch.exp(output)

                    test_loss += batch_loss.item()

                    top_prob, top_class = prob.topk(1, dim=1)
                    equals = top_class == labels_test.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"\nEpoch: {epoch+1}/{args.epoch}, Step {steps}")
            print(f"Training Loss: {running_loss / print_by}")
            print(f"Testing Loss: {test_loss / len(validloader)}")
            print("Test Accuracy: {:.2f}%".format(accuracy / len(validloader)*100))
            running_loss = 0
            model.train()

#training finished, print final testing accuracy
print("\n\n\nTraining Complete. Printing final accuracy on testing dataset.\n\n\n")
test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs_test, labels_test in testloader:
        inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
        output = model.forward(inputs_test)
        batch_loss = loss_criteria(output, labels_test)
        prob = torch.exp(output)

        test_loss += batch_loss.item()

        top_prob, top_class = prob.topk(1, dim=1)
        equals = top_class == labels_test.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Testing Loss: {test_loss / len(testloader)}")
print("Test Accuracy: {:.2f}%".format(accuracy / len(testloader)*100))

#save checkpoint
save_loc = args.save_dir

torch.save({
    'model_state_dict' : model.state_dict(),
    'optimizer_state_dict' : optimizer.state_dict(),
    'hidden_layers' : args.hidden,
    'model_name' : args.model_name,
    'class_to_idx' : class_to_idx
}, save_loc)

