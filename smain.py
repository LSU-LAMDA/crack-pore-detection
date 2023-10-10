import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import time
from utils import Dataset, prepare_masks, get_preprocessing, visualize

if __name__ == '__main__':
    
    for iter in range(1):
        print("$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(f"$$$$$ iter = {iter} $$$$$")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$")
        since = time.time()

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        torch.cuda.empty_cache()

        # Datasets to choose from [... , segzeiss, ...]
        dataset = 'segzeiss'
        data_dir = "./data/" + dataset

        x_trn_dir = data_dir + "/train/images"
        y_trn_dir = data_dir + "/train/masks"

        x_val_dir = data_dir + "/val/images"
        y_val_dir = data_dir + "/val/masks"

        x_tst_dir = data_dir + "/test/images"
        y_tst_dir = data_dir + "/test/masks"

        # Batch size for training (change depending on how much memory you have)
        batch_size = 8

        # Number of epochs to train for
        num_epochs = 2

        ENCODER = "resnet18"
        ENCODER_WEIGHTS = "imagenet"
        CLASSES = ["crack", "pore"]
        ACTIVATION = 'sigmoid'

        # phase
        train = False

        # visualization
        plot = True

        # smp.
        model = smp.Unet(
            # encoder_depth=3,
            encoder_name=ENCODER,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=ENCODER_WEIGHTS,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=len(CLASSES),                      # model output channels (number of classes in your dataset)
            activation= ACTIVATION,
            # decoder_channels=[256,128,64]
        )

        total_params = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        seghead_params = sum(p.numel() for p in model.segmentation_head.parameters())
        total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Encoder parameters: {encoder_params}")
        print(f"Decoder parameters: {decoder_params}")
        print(f"SegHead parameters: {seghead_params}")
        print(f"Total trainable parameters: {total_params_trainable // 1e6}m")

        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

        model = model.to(device)

        # Observe that all parameters are being optimized
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        train_dataset = Dataset(x_trn_dir,
                                y_trn_dir, 
                                augmentation=None, 
                                preprocessing=get_preprocessing(preprocessing_fn),
                                classes=CLASSES)

        valid_dataset = Dataset(x_val_dir, 
                                y_val_dir, 
                                augmentation=None, 
                                preprocessing=get_preprocessing(preprocessing_fn),
                                classes=CLASSES)


        trn_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        loss = smp.utils.losses.DiceLoss()
        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]

        optimizer = torch.optim.Adam([ 
            dict(params=model.parameters(), lr=0.0001),
        ])

        train_epoch = smp.utils.train.TrainEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            optimizer=optimizer,
            device=device,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            device=device,
            verbose=True,
        )

        if train:
            # train model for num_epochs epochs
            max_score = 0

            for i in range(0, num_epochs):
                
                print('\nEpoch: {}'.format(i))
                train_logs = train_epoch.run(trn_loader)
                valid_logs = valid_epoch.run(val_loader)
                
                # do something (save model, change lr, etc.)
                if max_score < valid_logs['iou_score']:
                    max_score = valid_logs['iou_score']
                    torch.save(model, f'./best_model{iter}.pth')
                    print('Model saved!')
                    
                if i == 25:
                    optimizer.param_groups[0]['lr'] = 1e-5
                    print('Decrease decoder learning rate to 1e-5!')                      

                if i == 50:
                    optimizer.param_groups[0]['lr'] = 1e-6
                    print('Decrease decoder learning rate to 1e-6!')  

                if i == 0:
                    log_path = f"report_iter{iter}.csv"
                    with open(log_path, 'w') as f:
                        f.write('time,iter,epoch,train_mIoU,valid_mIoU\n')

                with open(log_path, 'a') as f:
                    f.write(f"{time.time()-since}, {iter}, {i} , {train_logs['iou_score']}, {valid_logs['iou_score']}\n")


            torch.cuda.empty_cache()

        else:
            # Testing
            best_model = torch.load(f'./best_model{iter}.pth')

            # create test dataset
            test_dataset = Dataset(
                x_tst_dir, 
                y_tst_dir, 
                augmentation=None, 
                preprocessing=get_preprocessing(preprocessing_fn),
                classes=CLASSES,
            )

            test_dataloader = DataLoader(test_dataset)

            test_epoch = smp.utils.train.ValidEpoch(
                model=best_model,
                loss=loss,
                metrics=metrics,
                device=device,
            )

            logs = test_epoch.run(test_dataloader)

            test_dataset_vis = Dataset(
                x_tst_dir, y_tst_dir, 
                classes=CLASSES,
            )

            if plot:

                for i in range(40):
                    # n = np.random.choice(len(test_dataset))
                    n=i
                    
                    image_vis = test_dataset_vis[n][0].astype('uint8')
                    image, gt_mask = test_dataset[n]
                    
                    gt_mask = gt_mask.squeeze()
                    
                    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
                    pr_mask = best_model.predict(x_tensor)
                    pr_mask = (pr_mask.squeeze().cpu().detach().numpy().round())
                        
                    visualize(
                        image=image_vis, 
                        ground_truth_mask=gt_mask[0]*1+gt_mask[1]*2, 
                        predicted_mask=pr_mask[0]*1+pr_mask[1]*2
                    )
                    print(f"pr_mask shape: {pr_mask.shape}.")

        print("Done!")
        time_elapsed = time.time()-since
        print(f"Total time: {time_elapsed // 60}m {time_elapsed % 60 :.2f}s")