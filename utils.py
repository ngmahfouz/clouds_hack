import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

def initialize_model(model_name="resnet", num_classes=-1, feature_extract=True, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features

        #model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        
        #model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        #num_ftrs = 25088
        
        #model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        
        #model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        
        #model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        
        #model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        
        #model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    if model_name == "vgg":
        del model_ft.classifier[6]
        feature_extractor = model_ft
    else:
        modules=list(model_ft.children())[:-1]
        feature_extractor = nn.Sequential(*modules)

    return feature_extractor, input_size, num_ftrs


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def optim_step(optimizer, optim_type, step, i):
    if ("extra" in optim_type.lower() and (step % 2 == 0 or i == 0)):
        optimizer.extrapolation()
    else:
        optimizer.step()

    return optimizer

def fft(img):
    img = torch.from_numpy(img).to(torch.float32)
    img = torch.cat([img[:, :, None], torch.zeros_like(img[:, :, None])], dim=-1).to(torch.float32)
    fft = torch.fft(img, 2)
    fft_r = fft[:, :, 0]
    fft_i = fft[:, :, 1]
    ffn =torch.sqrt(fft_r ** 2 + fft_i ** 2)
    c = 255.0 / torch.log(1 + ffn.max())
    ffn = c * torch.log(1 + ffn).numpy()
    return ffn

def to_0_1(arr_or_tensor):
    """scales a tensor/array to [0, 1] values:
    (x - min(x)) / (max(x) - min(x))
    Args:
        arr_or_tensor (torch.Tensor or np.array): input tensor to scale
    Returns:
        torch.Tensor or np.array: scaled tensor
    """

    return (arr_or_tensor - arr_or_tensor.min()) / (
        arr_or_tensor.max() - arr_or_tensor.min()
    )
