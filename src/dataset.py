import cv2
import numpy as np
import os
try:
    import albumentations as aug
    from albumentations import (HorizontalFlip,VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise,RandomRotate90,Transpose,RandomBrightnessContrast, RandomCrop)
    from albumentations.pytorch import ToTensor
except:
    os.system(f"""pip install albumentations""")  
    import albumentations as aug
    from albumentations import (HorizontalFlip,VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise,RandomRotate90,Transpose,RandomBrightnessContrast, RandomCrop)
    from albumentations.pytorch import ToTensor  
from torch.utils.data import DataLoader, Dataset
try:
    import tifffile as tiff
except:
    os.system(f"""pip install -qq tifffile""")
    import tifffile as tiff

class EndoDataset(Dataset):
    def __init__(self, phase,shape = 512,train_size = 400):
        self.transforms = get_transforms(phase)
        self.phase = phase
        self.shape = shape
        self.train_size = train_size

    def __getitem__(self, idx):
        if self.phase == 'val':
            idx = idx + self.train_size
        mask = tiff.imread('Input/ead2020_semantic_segmentation/masks_ead2020/EAD2020_semantic_'+"{:05d}".format(idx)+'.tif')
        img = cv2.imread('Input/ead2020_semantic_segmentation/images_ead2020/EAD2020_semantic_'+"{:05d}".format(idx)+'.jpg')
        img = cv2.resize(img,(self.shape,self.shape))
        mask_re = np.zeros((5,self.shape,self.shape))
        for i in range(5):
            mask_re[i] = cv2.resize(mask[i],(self.shape,self.shape),interpolation = cv2.INTER_NEAREST)
        mask = mask_re.transpose(1,2,0)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        mask = mask[0].permute(2, 0, 1)
        return img, mask/255

    def __len__(self):
        if self.phase == 'train':
            return self.train_size
        else:
            return 474-self.train_size

def get_transforms(phase):
    list_transforms = []
    if phase == "train":
        list_transforms.extend([
     aug.Flip(),
    #  aug.OneOf([
    #         aug.CLAHE(clip_limit=2, p=.5),
    #         aug.IAASharpen(p=.25),
    #         ], p=0.35),
     aug.OneOf([
         aug.RandomContrast(),
         aug.RandomGamma(),
         aug.RandomBrightness(),
         ], p=1),
    #  aug.OneOf([
    #      aug.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    #      aug.GridDistortion(),
    #      aug.OpticalDistortion(distort_limit=2, shift_limit=0.5),
    #      ], p=0.3),
     aug.ShiftScaleRotate(rotate_limit=90),
     aug.OneOf([
            aug.GaussNoise(p=.35),
            ], p=.5),
    #  aug.Cutout(num_holes=3, p=.25),
    ])
    list_transforms.extend(
        [
            Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms

def provider(phase, batch_size=8, num_workers=0,):
    '''Returns dataloader for the model training'''
    image_dataset = EndoDataset(phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,   
    )

    return dataloader