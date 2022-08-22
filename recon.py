import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import numpy as np
import time
import os
import sys
from tqdm import tqdm as tqdm
from utils import DatasetRecon, splitter, patcher, get_preprocessing

if __name__ == '__main__':

    since = time.time()

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.cuda.empty_cache()

    # Datasets to choose from [... , segzeiss, ...]
    dataset = 'segzeiss'
    data_dir = "./data/" + dataset

    x_tst_dir = data_dir + "/test/images"
    y_tst_dir = data_dir + "/test/masks"

    x_rcn_dir = data_dir + "/recon_test/original"
    save_dir = data_dir + "/recon_test/segmented"

    ids = os.listdir(x_rcn_dir)

    CLASSES = ["crack", "pore"]

    model = torch.load(f'./best_model.pth')
    model = model.to(device)
    
    X = splitter(x_rcn_dir,patch_size=256,save=False,extend=True,plot=False,channels=3)

    for idx, patchset in enumerate(X[0]):

        print("$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(f"$$$$$ ImageNo = {idx} $$$$$")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$")
        
        ENCODER = "resnet18"
        ENCODER_WEIGHTS = "imagenet"

        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

        recon_dataset = DatasetRecon(patchset,preprocessing=get_preprocessing(preprocessing_fn))
        recon_dataloader = DataLoader(recon_dataset,batch_size=1)

        pr_mask = torch.zeros((64,len(CLASSES),256,256))

        count=0
        with tqdm(recon_dataloader, desc="Reconstruction", file=sys.stdout, disable=False) as iterator:
            for x in iterator:
                x = x.to(device)
                with torch.no_grad():
                    y = model.predict(x)
                pr_mask[count] = y
                count += 1

        pr_mask_np = pr_mask.permute(0,2,3,1).cpu().detach().numpy().round()
        pr_mask_flat = pr_mask_np[:,:,:,0]*100 + pr_mask_np[:,:,:,1]*200
        pr_mask_flat = np.expand_dims(pr_mask_flat, axis=-1)
        pr_mask = patcher(pr_mask_flat*1,save_dir=save_dir, slice_num=ids[idx]+"_mask", save=True, plot=False)


    print("Done!")
    time_elapsed = time.time()-since
    print(f"Total time: {time_elapsed // 60}m {time_elapsed % 60 :.2f}s")