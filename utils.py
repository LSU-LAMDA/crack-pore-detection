import os
import numpy as np
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from torch.utils.data import Dataset as BaseDataset

import albumentations as albu



class Dataset(BaseDataset):
    """Zeiss Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    
    CLASSES = ['bgbase', 'crack', 'pore']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. pores)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)


class DatasetRecon(BaseDataset):
    """Zeiss Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['bgbase', 'crack', 'pore']
    # CLASSES = ['bg', 'base', 'crack']
    
    def __init__(
            self, 
            images, 
            preprocessing=None,
    ):
        self.images = images
        
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = self.images[i]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
            
        return image
        
    def __len__(self):
        return len(self.images)


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap="gray")
    plt.show()

    
def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(256, 256)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)




def prepare_masks(masks_dir,save_dir=""):
    """
    This function converts marked regions of each manually segmented image into
    one single segmentation mask.
    - It is assumed the color of all marked regions is magenta [255,0,255]
    """
    
    org_dir = os.path.join(masks_dir, "original")
    ids = os.listdir(org_dir)

    classes = ['bg','cracks']

    # 0 is the background/base material
    # class_values = [1, 2]
    class_values = [0,2]

    images_fps = np.ones([len(classes)], dtype=list)

    for i,cls in enumerate(classes):
        class_dir = os.path.join(masks_dir, cls)
        print(class_dir)
        images_fps[i] = [os.path.join(class_dir, image_id) for image_id in ids]
    

    for j,imgid in enumerate(ids):
        mask = cv2.imread(images_fps[0][j])[:,:,0]*0+1
        for i,cls in enumerate(classes):
            img = cv2.imread(images_fps[i][j])
            ind = (img == [255, 0, 255])[:,:,0]
            mask[ind] = class_values[i]*0

        mask_name = imgid
        cv2.imwrite(os.path.join(save_dir, mask_name), mask)

    return None


def splitter(image_dir, patch_size, save_dir="", save=True, extend=False, plot=True, channels=1):
    """
    This function splits the original image to square patches.
    """

    ids = os.listdir(image_dir)

    images_fps = [os.path.join(image_dir, image_id) for image_id in ids]
    images = []

    # for fps in images_fps:
    for i,imgid in enumerate(images_fps):
        oimg = cv2.imread(imgid)
        h, w, c = oimg.shape

        count=0
        print("Saber: "+str(oimg.shape))

        if plot:
            fig,ax = plt.subplots(1)
            ax.imshow(oimg,cmap='gray', vmin=0, vmax=255)

        if (h % patch_size > 0 or w % patch_size > 0):
            if extend:
                print("!!! WARNING !!!: Original images size not dividable by patch size.")
                print("extend=True ==> EXTENDING THE BACKGROUND!")
                n0 = int(np.ceil(h/patch_size))
                n1 = int(np.ceil(w/patch_size))
                max_patches = n0*n1
            else:
                print("!!! WARNING !!!: Original images size not dividable by patch size.")
                n0 = int(h/patch_size)
                n1 = int(w/patch_size)
        else:
            n0 = h//patch_size
            n1 = w//patch_size
        max_patches = n0*n1
        hnew = n0*patch_size
        wnew = n1*patch_size
        X = np.zeros((max_patches,patch_size,patch_size,channels),dtype=np.uint8)

        for j in range(0,hnew,patch_size):
            for k in range(0,wnew,patch_size):
                upperleft_x = k
                upperleft_y = j
                img = oimg[upperleft_x:upperleft_x+patch_size, upperleft_y:upperleft_y+patch_size]

                d0 = img.shape[0]
                d1 = img.shape[1]

                X[count,0:d0,0:d1]=img[:,:,0:channels]
                count+=1
                if plot:
                    rect=patches.Rectangle((upperleft_y,upperleft_x),patch_size,patch_size,linewidth=4,
                                    edgecolor='w',facecolor="none")

                    ax.add_patch(rect)

                if save:
                    cv2.imwrite(os.path.join(save_dir,ids[i] + str(j//patch_size) + str(k//patch_size) +'.png'), X[count-1,:,:,0:channels])
        X=X[:count]

        print("Count: "+str(count))

        images.append(X)

        if plot:
            plt.show() 

    return images,(n0,n1)



def patcher(patches, save_dir="",slice_num="0", save=True, plot=True):
    """
    This function connects the square patches together to make a single image.
    """

    # Assuming patches are squared np arrays
    patch_count = patches.shape[0]

    # Assuming original image is square
    row_count = int(np.sqrt(patch_count))
    col_count = row_count

    nx = row_count
    ny = col_count
    col=[]
    for j in range(ny):
        col.append(np.concatenate(patches[j*nx:(j+1)*nx,:,:,:],axis=0))
        
    patched=np.concatenate((col[:]),axis=1)

    if save:
        if patches.shape[-1] == 2:
            cv2.imwrite(os.path.join(save_dir,slice_num +'0.png'), patched[:,:,0])
            cv2.imwrite(os.path.join(save_dir,slice_num +'1.png'), patched[:,:,1])
        else:
            cv2.imwrite(os.path.join(save_dir,slice_num +'.png'), patched)

    if plot:
        fig,ax=plt.subplots(1)
        ax.imshow(patched, cmap='gray', vmin=0, vmax=255)
        plt.title("Reconstructed image")
        plt.show() 

    return patched


if __name__ == "__main__":

    dataset = "segzeiss"
    phase = "train"
    unprocessed_dir = "./data/segzeiss/unprocessed/" + phase
    prepare_masks(unprocessed_dir, save_dir="./data/segzeiss/" + phase + "_mask")

    splitter("./data/segzeiss/unprocessed/"+ phase + "/original", 256, save_dir= "./data/segzeiss/"+ phase + "/images", extend=True, save=True, channels=3)
    splitter("./data/segzeiss/"+ phase + "_mask", 256, save_dir= "./data/segzeiss/"+ phase + "/masks", extend=True, save=True, channels=3)