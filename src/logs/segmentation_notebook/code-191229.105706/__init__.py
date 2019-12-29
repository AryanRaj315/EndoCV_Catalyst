import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision
from torchvision import datasets, models, transforms, utils
import pandas as pd
from skimage import io, transform
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm_notebook as tqdm
import albumentations as aug
from albumentations import (HorizontalFlip,VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise,RandomRotate90,Transpose,RandomBrightnessContrast, RandomCrop)
from albumentations.pytorch import ToTensor
plt.ion()   # interactive mode

seed = 23
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True