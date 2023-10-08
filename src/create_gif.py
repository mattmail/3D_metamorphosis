import imageio
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2

"""images = []

target = nib.load("/home/matthis/datasets/target_final.nii.gz").get_fdata()
deformed = nib.load("deformed_im.nii.gz").get_fdata()
source_file = "BraTS2021_00036"
source_img = nib.load("/home/matthis/datasets/brats_preproc/"+ source_file + "/" + source_file + "t1.nii.gz").get_fdata()

for i in range(target.shape[-1]):
    image = np.concatenate([source_img[:,:,i], target[:,:,i], deformed[:,:,i]], axis=1)
    images.append(image)

imageio.mimsave('deformation_mu0-1.mp4', images)"""

def create_heatmap_vid():
    target_img = nib.load("/home/matthis/Nextcloud/templates/T1_brain.nii").get_fdata().squeeze()
    target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min())
    target = target_img[:, ::-1, :-5]
    map_vm = nib.load('/home/matthis/Nextcloud/voxelmorph/scripts/torch/prob_map.nii.gz').get_fdata()
    map_vm = np.pad(map_vm, ((10,10),(10,10),(0,0)), "constant", constant_values=0)[:,:,:-2]
    map_meta = nib.load('/home/matthis/Nextcloud/3D_metamorphoses/src/prob_map.nii.gz').get_fdata()
    map_meta = np.pad(map_meta, ((10, 10), (10, 10), (0, 0)), "constant", constant_values=0)
    images = []
    for i in range(target.shape[2]):
        image_vm = add_heatmap(target[:,:,i], map_vm[:,:,i])
        image_meta = add_heatmap(target[:,:,i], map_meta[:,:,i])
        images.append(np.concatenate( (image_vm, image_meta )))
    imageio.mimsave('atlas_comparaison.mp4', images)


def add_heatmap(image, map):
    image_rgb = np.stack([image, image, image], axis=2).astype(np.float32)
    map = cv2.normalize(map, None, alpha=0., beta=255., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    map = cv2.applyColorMap(map, cv2.COLORMAP_HOT)
    map = map.astype(np.float32)/255.
    image_rgb = cv2.addWeighted(image_rgb, 0.5, map, 0.5, 0)
    return cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

create_heatmap_vid()