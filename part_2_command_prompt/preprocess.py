import torch
from torchvision import datasets, transforms, models
import PIL
import numpy as np

def create_datasets(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transform = transforms.Compose([transforms.RandomRotation(30), 
                                          transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    valid_transform = transforms.Compose([transforms.Resize(255), 
                                            transforms.CenterCrop(224), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize(255), 
                                            transforms.CenterCrop(224), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, valid_transform)
    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 32, shuffle = False)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = False)

    return trainloader, validloader, testloader, train_dataset.class_to_idx

def prepare_image(image_path):
    with PIL.Image.open(image_path) as im:
        #resize image to x by 256
        im.thumbnail((512, 256))
        
        #crop to center 224x224
        crop_factor_hor = (im.size[0] - 224) / 2
        crop_factor_vert = (im.size[1] - 224) / 2
        im_cropped = im.crop((crop_factor_hor, crop_factor_vert, crop_factor_hor + 224, crop_factor_vert + 224))
        
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        
        np_im = np.array(im_cropped)
        flattened_im = np_im / float(255)
        norm_im = (flattened_im - means) / stds
        transposed_im = norm_im.transpose(2, 0, 1)
        
        return torch.from_numpy(transposed_im)
