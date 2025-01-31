import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
from utils.tools import *
from dataset.Sen2Fire_Dataset import Sen2FireDataSet
from model.Networks import unet
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

name_classes = np.array(['non-fire','fire'], dtype=str)
epsilon = 1e-14

def get_arguments():

    parser = argparse.ArgumentParser()
    
    #dataset
    parser.add_argument("--data_dir", type=str, default='/mimer/NOBACKUP/groups/alvis_cvl/yonghao/Data/Sen2Fire/',
                        help="dataset path.")
    parser.add_argument("--train_list", type=str, default='./dataset/train.txt',
                        help="training list file.")
    parser.add_argument("--val_list", type=str, default='./dataset/val.txt',
                        help="val list file.")         
    parser.add_argument("--test_list", type=str, default='./dataset/test.txt',
                        help="test list file.")               
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")         
    parser.add_argument("--mode", type=int, default=10,
                        help="input type (0-all_bands, 1-all_bands_aerosol,...).")           

    #network
    parser.add_argument("--batch_size", type=int, default=8,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers for multithread dataloading.")
    
    #result
    parser.add_argument("--restore_from", type=str, default='./Exp/input_swir_aerosol/weight_10.0_time0104_1120/best_model.pth',
                        help="trained model.")
    parser.add_argument("--snapshot_dir", type=str, default='./Map/',
                        help="where to save detection results.")

    return parser.parse_args()

modename = ['all_bands',                        #0
            'all_bands_aerosol',                #1
            'rgb',                              #2
            'rgb_aerosol',                      #3
            'swir',                             #4
            'swir_aerosol',                     #5
            'nbr',                              #6
            'nbr_aerosol',                      #7   
            'ndvi',                             #8
            'ndvi_aerosol',                     #9 
            'rgb_swir_nbr_ndvi',                #10
            'rgb_swir_nbr_ndvi_aerosol',]       #11

def main():

    args = get_arguments()
    snapshot_dir = args.snapshot_dir+args.restore_from.split('/')[2]+'/'
    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)
    
    input_size = (512, 512)

    cudnn.enabled = True
    cudnn.benchmark = True
    
    # Create network
    if args.mode == 0:
        model = unet(n_classes=args.num_classes, n_channels=12)
    elif args.mode == 1:
        model = unet(n_classes=args.num_classes, n_channels=13)
    elif args.mode == 2 or args.mode == 4 or args.mode == 6 or args.mode == 8:
        model = unet(n_classes=args.num_classes, n_channels=3)
    elif args.mode == 3 or args.mode == 5 or args.mode == 7 or args.mode == 9:       
        model = unet(n_classes=args.num_classes, n_channels=4)
    elif args.mode == 10:       
        model = unet(n_classes=args.num_classes, n_channels=6)
    elif args.mode == 11:       
        model = unet(n_classes=args.num_classes, n_channels=7)

    saved_state_dict = torch.load(args.restore_from)  
    model.load_state_dict(saved_state_dict)
    model.eval()
    model = model.to(device)
    
    test_loader = data.DataLoader(
                    Sen2FireDataSet(args.data_dir, args.test_list,mode=args.mode),
                    batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # interpolation for the probability maps and labels 
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
        
    tbar = tqdm(test_loader)
    for _, batch in enumerate(tbar):  
        image, _,_,name = batch        
        image = image.float().to(device)
        
        with torch.no_grad():
            pred = model(image)
        _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy().astype('uint8')         

        patch_name = name[0].split('/')[1]
        patch_path = os.path.join(snapshot_dir, patch_name)
        np.savez_compressed(patch_path, label=pred)

    # Path to the directory containing test image patches
    image_dir = os.path.join(args.data_dir,"scene4")
    n_row = 21
    n_col = 24
    # Image size and patch information
    patch_size = (512, 512)
    overlap = 128
    original_image_size = (3, n_row * (patch_size[0] - overlap) + overlap, n_col * (patch_size[1] - overlap) + overlap)

    # Initialize an empty array to store the reconstructed image
    reconstructed_rgb = np.zeros(original_image_size)
    reconstructed_label = np.zeros(original_image_size[1:])
    reconstructed_pred = np.zeros(original_image_size[1:])

    # Iterate through rows and columns of patches
    for row in tqdm(range(1, n_row + 1)):  
        for col in range(1, n_col + 1): 
            # Load the patch
            patch_name = f"scene_4_patch_{row}_{col}.npz"
            patch_path = os.path.join(image_dir, patch_name)
            patch_data = np.load(patch_path)['image']
            patch_gt = np.load(patch_path)['label']
            pred_path = os.path.join(snapshot_dir, patch_name)
            patch_pred = np.load(pred_path)['label']

            # Calculate the starting indices for this patch in the original image
            start_row = (row - 1) * (patch_size[0] - overlap)
            start_col = (col - 1) * (patch_size[1] - overlap)

            # Update the reconstructed image with the patch
            reconstructed_rgb[:, start_row:start_row + patch_size[0], start_col:start_col + patch_size[1]] = patch_data[[3,2,1],:,:]
            reconstructed_label[start_row:start_row + patch_size[0], start_col:start_col + patch_size[1]] = patch_gt

            if (col==0)and(row==0):            
                reconstructed_pred[start_row:start_row + patch_size[0], start_col:start_col + patch_size[1]] = patch_pred
            elif (col==0)and(row!=0):
                reconstructed_pred[start_row+int(overlap/2):start_row + patch_size[0], start_col:start_col + patch_size[1]] = patch_pred[int(overlap/2):,:]
            elif (col!=0)and(row==0):
                reconstructed_pred[start_row:start_row + patch_size[0], start_col+int(overlap/2):start_col + patch_size[1]] = patch_pred[:,int(overlap/2):]
            elif (col!=0)and(row!=0):
                reconstructed_pred[start_row+int(overlap/2):start_row + patch_size[0], start_col+int(overlap/2):start_col + patch_size[1]] = patch_pred[int(overlap/2):,int(overlap/2):]

    # Apply ggplot style
    plt.style.use('ggplot')
    cmap = ListedColormap(['white', 'red'])

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    axs[0].imshow(reconstructed_rgb.transpose(1, 2, 0)/1500.)
    axs[0].axis('off')
    axs[0].set_title('RGB image', fontsize=12)

    axs[1].imshow(reconstructed_rgb.transpose(1, 2, 0)/1500., alpha=0.6)
    axs[1].imshow(reconstructed_pred, cmap=cmap, alpha=0.7)
    axs[1].axis('off')
    axs[1].set_title('Detection', fontsize=12)

    axs[2].imshow(reconstructed_rgb.transpose(1, 2, 0)/1500., alpha=0.6)
    axs[2].imshow(reconstructed_label, cmap=cmap, alpha=0.7)
    axs[2].axis('off')
    axs[2].set_title('Label', fontsize=12)
    legend_labels = ['Non-fire', 'Fire']
    plt.legend(handles=[plt.Rectangle((0,0),1,1,facecolor=cmap(i),edgecolor='black') for i in range(2)], labels=legend_labels, fontsize=12, frameon=False, bbox_to_anchor=(1.04, 0), loc="lower left")
    plt.tight_layout()

    save_path = os.path.join(snapshot_dir, "Fig_5_map.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    
if __name__ == '__main__':
    main()
