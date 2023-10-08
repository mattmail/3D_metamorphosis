from time import time

import torch
import numpy as np

from prepare_data import load_brats_2021
from models import meta_model, meta_model_local, meta_model_local_sharp, metamorphoses
from train import train_opt



if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    use_segmentation = True

    n_epoch = 10000
    l = 15
    L2_weight = .5
    lamda = 1e-7
    v_weight = lamda / l
    z_weight = 0. / l
    mu = 0.01
    batch_size = 1
    sigma = 4.
    debug = False

    _, test_loader, target_img, config = load_brats_2021(device, batch_size, get_ventricles=True)

    config["debug"] = debug
    config['batch_size'] = batch_size
    config['n_epoch'] = n_epoch
    config['local_reg'] = use_segmentation
    config["plot_epoch"] = 1
    config["L2_weight"] = L2_weight
    config['v_weight'] = v_weight
    config["z_weight"] = z_weight
    config['inv_weight'] = 0.005
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
    model = metamorphoses(l, z0.shape, device, 31, sigma, mu, z0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e0,
                                 weight_decay=0)

    L2_norm_list = []
    L2_no_tumor_list = []
    L2_def_list = []
    num_folds = []
    dice_ventricles = []

    for i,source in enumerate(test_loader):
        if config['debug'] and i==5:
            break
        model = metamorphoses(l, z0.shape, device, 31, sigma, mu, z0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                     weight_decay=1e-8)
        L2_norm, L2_no_tumor, L2_def, dice, folds = train_opt(model, source, target_img, optimizer, config)
        L2_norm_list.append(L2_norm.detach().cpu())
        L2_no_tumor_list.append(L2_no_tumor.detach().cpu())
        L2_def_list.append(L2_def.detach().cpu())
        dice_ventricles.append(dice.detach().cpu())
        num_folds.append(folds.detach().cpu())
        print("\nImage Number ", i)
        print("L2 loss:", L2_norm.detach().cpu().item(), "Dice:", dice.detach().cpu().item(), "Fold number:", folds.detach().cpu().item())

    print("Validation L2 loss: %f" % (sum(L2_norm_list) / len(test_loader)),
          "std: %f" % (np.array(L2_norm_list).std()))
    print("Validation L2 loss no tumor: %f" % (sum(L2_no_tumor_list) / len(test_loader)),
          "std: %f" % (np.array(L2_no_tumor_list).std()))
    print("Validation L2 deformation only: %f" % (sum(L2_def_list) / len(test_loader)),
          "std: %f" % (np.array(L2_def_list).std()))
    print("Average fold number:", sum(num_folds) / len(num_folds), "std: %f" % (np.array(num_folds).std()))
    print("Validation dice:", sum(dice_ventricles) / len(dice_ventricles),
          "std: %f" % (np.array(dice_ventricles).std()))




