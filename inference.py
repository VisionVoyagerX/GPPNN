from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary

from data_loader.DataLoader import DIV2K, GaoFen2, Sev2Mod, WV3, GaoFen2panformer
from models.GPPNN import GPPNN
from utils import *
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Prepare device
    # TODO add more code for server
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize DataLoader
    train_dataset = WV3(
        Path("/home/ubuntu/project/Data/WorldView3/train/train_wv3-001.h5"))  # transforms=[(RandomHorizontalFlip(1), 0.3), (RandomVerticalFlip(1), 0.3)]
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=4, shuffle=True, drop_last=True)

    validation_dataset = WV3(
        Path("/home/ubuntu/project/Data/WorldView3/val/valid_wv3.h5"))
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=1, shuffle=True)

    test_dataset = WV3(
        Path("/home/ubuntu/project/Data/WorldView3/drive-download-20230627T115841Z-001/test_wv3_multiExm1.h5"))
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False)

    # Initialize Model, optimizer, criterion and metrics
    model = GPPNN(8, 1, 64, 8, mslr_mean=train_dataset.mslr_mean.to(device), mslr_std=train_dataset.mslr_std.to(device), pan_mean=train_dataset.pan_mean.to(device),
                  pan_std=train_dataset.pan_std.to(device)).to(device)

    optimizer = Adam(model.parameters(), lr=5e-5, weight_decay=0)

    criterion = L1Loss().to(device)

    metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    val_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    test_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    tr_report_loss = 0
    val_report_loss = 0
    test_report_loss = 0
    tr_metrics = []
    val_metrics = []
    test_metrics = []
    best_eval_psnr = 0
    best_test_psnr = 0
    current_daytime = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    steps = 250000
    save_interval = 1000
    report_interval = 50
    test_intervals = [40000, 60000, 100000,
                      140000, 160000, 200000]
    evaluation_interval = [40000, 60000, 100000,
                           140000, 160000, 200000]
    continue_from_checkpoint = True

    val_steps = 100

    # Model summary
    summary(model, [(1, 1, 256, 256), (1, 8, 64, 64)],
            dtypes=[torch.float32, torch.float32])

    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    lr_decay_intervals = 50000

    # load checkpoint
    if continue_from_checkpoint:
        tr_metrics, val_metrics = load_checkpoint(torch.load(
            'checkpoints/gppnn_WV3/gppnn_WV3_2023_07_27-08_11_42.pth.tar'), model, optimizer, tr_metrics, val_metrics)
        print('Model Loaded ...')

    def scaleMinMax(x):
        return ((x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)))

    idx = 14
    # evaluation mode
    model.eval()
    with torch.no_grad():
        test_iterator = iter(test_loader)
        for i, (pan, mslr, mshr) in enumerate(test_iterator):
            if idx == i:
                # forward
                pan, mslr, mshr = pan.to(device), mslr.to(
                    device), mshr.to(device)
                mssr = model(pan, mslr)
                test_loss = criterion(mssr, mshr)
                test_metric = test_metric_collection.forward(mssr, mshr)
                test_report_loss += test_loss

                # compute metrics
                test_metric = test_metric_collection.compute()

                figure, axis = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
                axis[0].imshow((scaleMinMax(mslr.permute(0, 3, 2, 1).detach().cpu()[
                               0, ...].numpy())).astype(np.float32)[..., :3], cmap='viridis')
                axis[0].set_title('(a) LR')
                axis[0].axis("off")

                axis[1].imshow(pan.permute(0, 3, 2, 1).detach().cpu()[
                               0, ...], cmap='gray')
                axis[1].set_title('(b) PAN')
                axis[1].axis("off")

                axis[2].imshow((scaleMinMax(mssr.permute(0, 3, 2, 1).detach().cpu()[
                               0, ...].numpy())).astype(np.float32)[..., :3], cmap='viridis')
                axis[2].set_title(
                    f'(c) PanFormer {test_metric["psnr"]:.2f}dB/{test_metric["ssim"]:.4f}')
                axis[2].axis("off")

                axis[3].imshow((scaleMinMax(mshr.permute(0, 3, 2, 1).detach().cpu()[
                               0, ...].numpy())).astype(np.float32)[..., :3], cmap='viridis')
                axis[3].set_title('(d) GT')
                axis[3].axis("off")

                plt.savefig('results/Images_WV3.png')

                mslr = mslr.permute(0, 3, 2, 1).detach().cpu().numpy()
                pan = pan.permute(0, 3, 2, 1).detach().cpu().numpy()
                mssr = mssr.permute(0, 3, 2, 1).detach().cpu().numpy()
                gt = mshr.permute(0, 3, 2, 1).detach().cpu().numpy()

                np.savez('results/img_array_WV3.npz', mslr=mslr,
                         pan=pan, mssr=mssr, gt=gt)


if __name__ == '__main__':
    main()
