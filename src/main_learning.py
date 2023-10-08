import torch
import kornia.filters as flt
import numpy as np
import cv2
import os
from time import time
from models import meta_model, meta_model_local_sharp, shooting_model
from train import train_learning
from prepare_data import load_brats_2021, load_brats_2020


if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    use_segmentation = True

    n_epoch = 50
    l = 10
    L2_weight = .5
    lamda = 3e-10
    v_weight = lamda/l
    z_weight = lamda/l
    mu = 0.01
    batch_size = 1
    sigma = 4.
    debug = False

    train_loader, test_loader, target_img, config = load_brats_2021(device, batch_size, get_ventricles=True)

    config["debug"] = debug
    config['batch_size'] = batch_size
    config['n_epoch'] = n_epoch
    config['local_reg'] = use_segmentation
    config["plot_epoch"] = 1
    config["L2_weight"] = L2_weight
    config['v_weight'] = v_weight
    config["z_weight"] = z_weight
    config["l"] = l
    config["mu"] = mu
    config["device"] = device
    config["downsample"] = True


    if config["downsample"]:
        ndown = 2
        MNI_img_down = target_img[:, :, ::ndown, ::ndown, ::ndown]
        z0 = torch.zeros(MNI_img_down.shape, dtype=torch.float32)
    else:
        z0 = torch.zeros(target_img.shape, dtype=torch.float32)

    z0.requires_grad = True

    print("### Starting Metamorphoses ###")
    print("L2_weight=", L2_weight)
    print("z_weight=", z_weight)
    print("v_weight=", v_weight)
    print("n_epoch=", n_epoch)
    print("mu=", mu)
    t = time()
    """if use_segmentation:
        model = meta_model_local_sharp(l, z0.shape, device, 31, sigma, mu, z0).to(device)
    else:
        model = meta_model(l, z0.shape, device, 51, 2., mu, z0).to(device)"""
    model = shooting_model(l, z0.shape, device, 31, sigma, mu).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                  weight_decay=1e-8)

    train_learning(model, train_loader, test_loader, target_img, optimizer, config)





