import torch
import matplotlib.pyplot as plt
from utils import deform_image, create_meshgrid3d, make_grid
import numpy as np
from matplotlib import gridspec



from prepare_data import load_brats_2021
result_path="/home/matthis/Nextcloud/3D_metamorphoses/results/"
#result_path="/home/infres/maillard/3D_metamorphoses/results/"
#model_path = "meta_model_1024_1322/model.pt" #inv_loss = 0.0001
#model_path = "meta_model_1024_1648/model.pt" #inv_loss = 0.0001 , sigma=10
#model_path = "meta_model_1020_1003/model.pt" #inv_loss = 0.005
#model_path = "meta_model_0530_1148/model.pt" #results l4e-4
#model_path = "meta_model_1025_1716/model.pt" #h=50
#model_path = "meta_model_1108_1306/model.pt"
model_path = "meta_model_1115_0925/model.pt"
model_path = "meta_model_1122_0939/model.pt"

device = "cuda:0"
model = torch.load(result_path + model_path, map_location=device)
model.device = device

list_files = ['00071']

torch.manual_seed(5)
train_loader, test_loader, target_img, _ = load_brats_2021(device, 1, get_ventricles=True, return_name=True)
target = target_img.to(device)
model.eval()
grid = make_grid(target_img.shape, 12).to(device)

with torch.no_grad():
    l=model.l
    for j, (source, name) in enumerate(train_loader):
        if name[0].split("_")[1] in list_files or True:
            slice=80
            source_img = source[:, 0].to(device)
            source_seg = source[:, 1].to(device)
            #source_map = source[:, 2]
            source_deformed, fields, grad, residuals, residuals_deformed = model(source_img, target, source_seg)
            id_grid = create_meshgrid3d(fields[0].shape[3], fields[0].shape[2], fields[0].shape[1], device)
            residuals_list = [torch.zeros(residuals[0].shape).to(device)]
            residuals_deformed = torch.zeros(residuals[0].shape).to(device)
            back_residuals = []
            back_tot_res = torch.zeros(residuals[0].shape).to(device)
            forward_list = [id_grid.clone()]
            back_list = [id_grid.clone()]
            mask_list = []
            for i in range(l):
                back_def = id_grid + fields[l - i - 1] / l
                pos_def = id_grid - fields[i]/l
                forward_list.append(deform_image(forward_list[i].permute(0, 4, 3, 2, 1), pos_def).permute(0, 4, 3, 2, 1))
                back_list.append(deform_image(back_list[i].permute(0, 4, 3, 2, 1), back_def).permute(0, 4, 3, 2, 1))
                if i > 0:
                    residuals_deformed = deform_image(residuals_list[-1], pos_def)
                    back_tot_res = deform_image(back_residuals[-1], back_def)
                residuals_list.append(residuals_deformed + residuals[i+1])
                back_residuals.append(back_tot_res + residuals[l-i])
                mask_list.append(deform_image(source_seg, forward_list[-1]))

            #plt.figure(figsize=(2,8))
            nrow = 3
            ncol = 9
            fig = plt.figure(figsize=(ncol + 1, nrow + 1))
            gs = gridspec.GridSpec(nrow, ncol,
                                   wspace=0.0, hspace=0.0,
                                   top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                   left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
            select = [3,7,11,14]
            ax = plt.subplot(gs[0,0])
            ax.imshow(source_img.detach().squeeze().cpu()[:,:,slice].t(), cmap="gray", vmin=0., vmax=1.)
            ax.axis("off")
            ax = plt.subplot(gs[2, 0])
            ax.imshow(grid.detach().squeeze().cpu()[:, :, slice].t(), cmap="gray", vmin=0., vmax=1.)
            ax.axis("off")
            ax = plt.subplot(gs[1, 0])
            ax.imshow(torch.zeros(source_img.detach().squeeze().cpu()[:,:,slice].shape), cmap="coolwarm", vmin=-1, vmax=1)
            ax.axis("off")
            for i in range(0,4):
                image = deform_image(source_img, forward_list[select[i]+1]) + model.mu**2 / l * residuals_list[select[i]+1] * mask_list[select[i]]
                deformed_grid = deform_image(grid, forward_list[select[i]+1])
                res_sum = model.mu**2 / l * residuals_list[select[i]+1] * mask_list[select[i]]
                ax = plt.subplot(gs[0, i+1])
                ax.imshow(image.detach().squeeze().cpu()[:,:,slice].t(), cmap="gray", vmin=0., vmax=1.)
                ax.axis("off")
                ax = plt.subplot(gs[2,i+1])
                ax.imshow(deformed_grid.detach().squeeze().cpu()[:, :, slice].t(), cmap="gray", vmin=0., vmax=1.)
                ax.axis("off")
                ax = plt.subplot(gs[1, i + 1])
                ax.imshow(res_sum.detach().squeeze().cpu()[:, :, slice].t(), cmap="coolwarm", vmin=-1, vmax=1)
                ax.axis("off")
            deformed_image = image.clone()
            saved_grid = deformed_grid.clone()
            for i in range(0,4):
                image = deform_image(deformed_image - model.mu**2 / l * residuals_list[l] * mask_list[l-1], back_list[select[i]+1]) + model.mu**2 / l * residuals_list[l-select[i]-1] * mask_list[l-select[i]-2]
                inv_residuals = deform_image(res_sum- model.mu**2 / l * residuals_list[l] * mask_list[l-1], back_list[select[i]+1]) + model.mu**2 / l * residuals_list[l-select[i]-1] * mask_list[l-select[i]-2]
                ax = plt.subplot(gs[0, i+5])
                ax.imshow(image.detach().squeeze().cpu()[:,:,slice].t(), cmap="gray", vmin=0., vmax=1.)
                ax.axis("off")
                deformed_grid = deform_image(saved_grid, back_list[select[i] + 1])
                ax = plt.subplot(gs[2, i+5])
                ax.imshow(deformed_grid.detach().squeeze().cpu()[:, :, slice].t(), cmap="gray", vmin=0., vmax=1.)
                ax.axis("off")
                ax = plt.subplot(gs[1, i + 5])
                im = ax.imshow(inv_residuals.detach().squeeze().cpu()[:, :, slice].t(), cmap="coolwarm", vmin=-1, vmax=1)
                ax.axis("off")
            #cbar = plt.colorbar(im, ax=ax)
            #fig.tight_layout()
            #plt.suptitle(name[0])
            plt.subplots_adjust(wspace=0., hspace=0.)
            plt.savefig("../results/inverse_composition.png")
            plt.show()
            plt.imshow(inv_residuals.detach().squeeze().cpu()[:, :, slice].t(), cmap="coolwarm", vmin=-1, vmax=1)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=20)
            plt.show()



