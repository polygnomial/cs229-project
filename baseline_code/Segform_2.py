from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
from transformers import TrainingArguments
import segmentation_models_pytorch as smp
import numpy as np
import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn
from utils.tools import *
import torchvision.transforms as transforms
from dataset.Sen2Fire_Dataset import Sen2FireDataSet
import random
from tqdm import tqdm
import sys
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

name_classes = np.array(['non-fire','fire'], dtype=str)
epsilon = 1e-14

def init_seeds(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 


def get_arguments():
    parser = argparse.ArgumentParser()
    
    #  Specify segmentation model type, so we can see the result from different model
    parser.add_argument("--model_type", type=str, default='segformer', 
                      choices=['unet', 'deeplabv3plus', 'pspnet', 'fpn', 'customcnn', 'encoderfeature', 'segformer'],
                      help="Type of segmentation model to use.")
    parser.add_argument("--encoder_name", type=str, default='mit_b0',
                      choices=['resnet50', 'resnet101', 'efficientnet-b4', 'timm-regnety_016', 'resnext50_32x4d', 'mit_b0'],
                      help="Name of encoder backbone to use.")
                      
    #dataset
    parser.add_argument("--freeze_epochs", type=int, default=3000,   
                    help="Number of steps (or epochs) before unfreezing encoder.")
    parser.add_argument("--data_dir", type=str, default='/Users/d5826/desktop/milestone/Sen2Fire/',
                        help="dataset path.")
    parser.add_argument("--train_list", type=str, default='./dataset/train.txt',
                        help="training list file.")
    parser.add_argument("--val_list", type=str, default='./dataset/val.txt',
                        help="val list file.")         
    parser.add_argument("--test_list", type=str, default='./dataset/test.txt',
                        help="test list file.")               
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")   
    # for this fine-tuning, use mode 4, the best model from the original paper   
    parser.add_argument("--mode", type=int, default=4,
                        help="input type (0-all_bands, 1-all_bands_aerosol,...).")           

    #network
    parser.add_argument("--batch_size", type=int, default=8,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="base learning rate.")
    parser.add_argument("--num_steps", type=int, default=5000,
                        help="number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=5000,
                        help="number of training steps for early stopping.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="regularisation parameter for L2-loss.")
    parser.add_argument("--weight", type=float, default=10,
                        help="ce weight.")
    # for fine-tuning, unfreeze step
    parser.add_argument("--unfreeze_step", type=int, default=2000,
                        help="Number of steps before unfreezing U Net Encoder weights.")
    
    # From feedback: Progressive unfreezing strategy
    parser.add_argument("--progressive_unfreeze", action='store_true',
                        help="Whether to progressively unfreeze encoder layers.")

    #result
    parser.add_argument("--snapshot_dir", type=str, default='./Exp/',
                        help="where to save snapshots of the model.")

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

mode_channel_counts = [12, 13, 3, 4, 3, 4, 3, 4, 3, 4, 6, 7]

# From feedback: Model selection method
def get_model(args):
    # this is the huggingface segformer
    print(f"num channels: {mode_channel_counts[args.mode]}")
    if args.model_type == 'segformer':
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", 
            num_channels=mode_channel_counts[args.mode], 
            num_labels=args.num_classes,
            ignore_mismatched_sizes=True
        )
        # # fix the dimension mismatch 
        # dims_to_add = mode_channel_counts[args.mode] - 3
        # model.segformer.encoder.weight = nn.Parameter(
        #     torch.cat((model.segformer.encoder.weight, torch.randn(1, dims_to_add, 256)), 1)
        # )
        
        return model
    else:
        # this is a combined U-Net structure but with segformer encoder
        if args.encoder_name == 'mit_b0':
            return smp.Unet(
                encoder_name=args.encoder_name,
                encoder_weights="imagenet",
                in_channels=mode_channel_counts[args.mode],
                classes=args.num_classes
            )
        # Use models from smp library
        if args.model_type == 'unet':
            return smp.Unet(
                encoder_name=args.encoder_name,
                encoder_weights="imagenet",
                in_channels=int(mode_channel_counts[args.mode]),
                classes=args.num_classes
            )
        elif args.model_type == 'deeplabv3plus':
            return smp.DeepLabV3Plus(
                encoder_name=args.encoder_name,
                encoder_weights="imagenet",
                in_channels=mode_channel_counts[args.mode],
                classes=args.num_classes
            )
        elif args.model_type == 'pspnet':
            return smp.PSPNet(
                encoder_name=args.encoder_name,
                encoder_weights="imagenet",
                in_channels=mode_channel_counts[args.mode],
                classes=args.num_classes
            )
        elif args.model_type == 'fpn':
            return smp.FPN(
                encoder_name=args.encoder_name,
                encoder_weights="imagenet",
                in_channels=mode_channel_counts[args.mode],
                classes=args.num_classes
            )

def main(args):
    snapshot_dir = args.snapshot_dir+'model_'+args.model_type+'_encoder_'+'_input_'+modename[args.mode]+'/weight_'+str(args.weight)+'_time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
        
    f = open(snapshot_dir+'Training_log.txt', 'w')

    input_size_train = (512, 512)

    cudnn.enabled = True
    cudnn.benchmark = True
    init_seeds()

    # Get selected model
    model = get_model(args)
    model = model.to(device)
    print(model)

    # Create normalization function to manually apply to images rather than as transform parameter
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Remove transform parameter
    train_loader = data.DataLoader(
                    Sen2FireDataSet(args.data_dir, args.train_list, max_iters=args.num_steps_stop*args.batch_size,
                    mode=args.mode),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = data.DataLoader(
                    Sen2FireDataSet(args.data_dir, args.val_list, mode=args.mode),
                    batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    test_loader = data.DataLoader(
                    Sen2FireDataSet(args.data_dir, args.test_list, mode=args.mode),
                    batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # interpolation for the probability maps and labels 
    interp = nn.Upsample(size=(input_size_train[1], input_size_train[0]), mode='bilinear')
    
    hist = np.zeros((args.num_steps_stop,4))
    F1_best = 0.    

    class_weights = [1, args.weight]
    L_seg = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    
    # Add Dice Loss and BCE Loss
    L_dice = smp.losses.DiceLoss(mode="binary")
    L_bce = nn.BCEWithLogitsLoss()

    # Determine parameter groups and freezing strategy based on model type
    if args.model_type == 'encoderfeature':
        # Freeze encoder, only train custom layers
        for param in model.encoder.parameters():
            param.requires_grad = False
            
        optimizer = optim.Adam([
            {"params": model.encoder.parameters(), "lr": 1e-5},
            {"params": model.custom_layers.parameters(), "lr": args.learning_rate}
        ], weight_decay=args.weight_decay)
    elif args.model_type == 'segformer':
        # For segformer, inilziation 
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        # For pretrained models, freeze encoder
        if hasattr(model, 'encoder'):
            for param in model.encoder.parameters():
                param.requires_grad = False
            
            optimizer = optim.Adam([
                {"params": model.encoder.parameters(), "lr": 1e-5},
                {"params": [p for n, p in model.named_parameters() if not n.startswith('encoder')], "lr": args.learning_rate}
            ], weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for batch_index, train_data in enumerate(train_loader):
        print(batch_index)
        if batch_index == args.num_steps_stop:
            break
        tem_time = time.time()
        adjust_learning_rate(optimizer, args.learning_rate, batch_index, args.num_steps)
        model.train()
        optimizer.zero_grad()
        
        patches, labels, _, _ = train_data
        # Manually apply normalization
        patches = patches.to(device)
        # Apply normalization to each image in the batch, only if mode is not 5
        if args.mode != 5:
            for i in range(patches.size(0)):
                patches[i] = normalize(patches[i])
          
        labels = labels.to(device).long()
        
        # Handle forward pass based on model type
        if args.model_type == 'encoderfeature':
            pred, features = model(patches)
        else:
            if args.model_type == 'segformer':
                outputs = model(patches)
                pred = outputs.logits  # logits
            else:
                pred = model(patches)
            
        pred_interp = interp(pred)
        
        # Combined loss: Cross-Entropy + Dice Loss
        L_seg_value = L_seg(pred_interp, labels)
        L_dice_value = L_dice(F.softmax(pred_interp, dim=1)[:, 1:], labels == 1)
        
        # Combined loss
        loss = 0.7 * L_seg_value + 0.3 * L_dice_value
        
        _, predict_labels = torch.max(pred_interp, 1)
        lbl_pred = predict_labels.detach().cpu().numpy()
        lbl_true = labels.detach().cpu().numpy()
        metrics_batch = []
        for lt, lp in zip(lbl_true, lbl_pred):
            _,_,mean_iu,_ = label_accuracy_score(lt, lp, n_class=args.num_classes)
            metrics_batch.append(mean_iu)                
        batch_miou = np.nanmean(metrics_batch, axis=0)  
        batch_oa = np.sum(lbl_pred==lbl_true)*1./len(lbl_true.reshape(-1))
                    
        hist[batch_index,0] = loss.item()
        hist[batch_index,1] = batch_oa
        hist[batch_index,2] = batch_miou
        
        loss.backward()
        optimizer.step()

        hist[batch_index,-1] = time.time() - tem_time

        if (batch_index+1) % 10 == 0: 
            print('Iter %d/%d Time: %.2f Batch_OA = %.2f Batch_mIoU = %.2f Loss = %.3f'%(batch_index+1,args.num_steps,10*np.mean(hist[batch_index-9:batch_index+1,-1]),np.mean(hist[batch_index-9:batch_index+1,1])*100,np.mean(hist[batch_index-9:batch_index+1,2])*100,np.mean(hist[batch_index-9:batch_index+1,0])))
            f.write('Iter %d/%d Time: %.2f Batch_OA = %.2f Batch_mIoU = %.2f Loss = %.3f\n'%(batch_index+1,args.num_steps,10*np.mean(hist[batch_index-9:batch_index+1,-1]),np.mean(hist[batch_index-9:batch_index+1,1])*100,np.mean(hist[batch_index-9:batch_index+1,2])*100,np.mean(hist[batch_index-9:batch_index+1,0])))
            f.flush() 
        
        # Handle unfreezing strategy
        if args.model_type not in ['customcnn']:
            # Progressive unfreezing
            if args.progressive_unfreeze:
                if batch_index == int(args.unfreeze_step * 0.25):
                    print("\nUnfreezing encoder layer 4...\n")
                    for param in model.encoder.layer4.parameters():
                        param.requires_grad = True
                
                elif batch_index == int(args.unfreeze_step * 0.5):
                    print("\nUnfreezing encoder layer 3...\n")
                    for param in model.encoder.layer3.parameters():
                        param.requires_grad = True
                
                elif batch_index == int(args.unfreeze_step * 0.75):
                    print("\nUnfreezing encoder layer 2...\n")
                    for param in model.encoder.layer2.parameters():
                        param.requires_grad = True
                
                elif batch_index == args.unfreeze_step:
                    print("\nUnfreezing all encoder layers...\n")
                    for param in model.encoder.parameters():
                        param.requires_grad = True
            # Regular unfreezing
            elif batch_index == args.unfreeze_step:
                print("\nUnfreezing encoder for fine-tuning...\n")
                if hasattr(model, 'encoder'):
                    for param in model.encoder.parameters():
                        param.requires_grad = True

        # evaluation per 500 iterations
        if (batch_index+1) % 500 == 0:            
            print('Validating..........')  
            f.write('Validating..........\n')  

            model.eval()
            TP_all = np.zeros((args.num_classes, 1))
            FP_all = np.zeros((args.num_classes, 1))
            TN_all = np.zeros((args.num_classes, 1))
            FN_all = np.zeros((args.num_classes, 1))
            n_valid_sample_all = 0
            F1 = np.zeros((args.num_classes, 1))
            IoU = np.zeros((args.num_classes, 1))
            
            tbar = tqdm(val_loader)
            for idx, batch in enumerate(tbar):  
                image, label, _, _ = batch
                label = label.squeeze().numpy()
                image = image.float().to(device)
                
                # Manually apply normalization, only if mode is not 5
                if args.mode != 5:
                    for i in range(image.size(0)):
                        image[i] = normalize(image[i])
                
                with torch.no_grad():
                    if args.model_type == 'encoderfeature':
                        pred, features = model(image)
                    else:
                        if args.model_type == 'segformer':
                            outputs = model(image)
                            pred = outputs.logits  # logits
                        else:
                            pred = model(image)

                _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
                pred = pred.squeeze().data.cpu().numpy()                       
                               
                TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1),label.reshape(-1),args.num_classes)
                TP_all += TP
                FP_all += FP
                TN_all += TN
                FN_all += FN
                n_valid_sample_all += n_valid_sample

            OA = np.sum(TP_all)*1.0 / n_valid_sample_all
            for i in range(args.num_classes):
                P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
                R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
                F1[i] = 2.0*P*R / (P + R + epsilon)
                IoU[i] = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + FN_all[i] + epsilon)
            
                if i==1:
                    print('===>' + name_classes[i] + ' Precision: %.2f'%(P * 100))
                    print('===>' + name_classes[i] + ' Recall: %.2f'%(R * 100))            
                    print('===>' + name_classes[i] + ' IoU: %.2f'%(IoU[i] * 100))              
                    print('===>' + name_classes[i] + ' F1: %.2f'%(F1[i] * 100))   
                    f.write('===>' + name_classes[i] + ' Precision: %.2f\n'%(P * 100))
                    f.write('===>' + name_classes[i] + ' Recall: %.2f\n'%(R * 100))            
                    f.write('===>' + name_classes[i] + ' IoU: %.2f\n'%(IoU[i] * 100))              
                    f.write('===>' + name_classes[i] + ' F1: %.2f\n'%(F1[i] * 100))   
                
            mF1 = np.mean(F1)   
            mIoU = np.mean(IoU)           
            print('===> mIoU: %.2f mean F1: %.2f OA: %.2f'%(mIoU*100,mF1*100,OA*100))
            f.write('===> mIoU: %.2f mean F1: %.2f OA: %.2f\n'%(mIoU*100,mF1*100,OA*100))
                
            if F1[1]>F1_best:
                F1_best = F1[1]
                # save the models        
                print('Save Model')   
                f.write('Save Model\n')   
                model_name = 'best_model.pth'
                print(f"saving model to {os.path.join(snapshot_dir, model_name)}")
                torch.save(model.state_dict(), os.path.join(snapshot_dir, model_name))
    
    saved_state_dict = torch.load(os.path.join(snapshot_dir, model_name))  
    model.load_state_dict(saved_state_dict)

    print('Testing..........')  
    f.write('Testing..........\n')  

    model.eval()
    TP_all = np.zeros((args.num_classes, 1))
    FP_all = np.zeros((args.num_classes, 1))
    TN_all = np.zeros((args.num_classes, 1))
    FN_all = np.zeros((args.num_classes, 1))
    n_valid_sample_all = 0
    F1 = np.zeros((args.num_classes, 1))
    IoU = np.zeros((args.num_classes, 1))
    
    tbar = tqdm(test_loader)
    for idx, batch in enumerate(tbar):  
        image, label, _, _ = batch
        label = label.squeeze().numpy()
        image = image.float().to(device)
        
        # Manually apply normalization, only if mode is not 5
        if args.mode != 5:
            for i in range(image.size(0)):
                image[i] = normalize(image[i])
        
        with torch.no_grad():
            if args.model_type == 'encoderfeature':
                pred, features = model(image)
            else:
                if args.model_type == 'segformer':
                    outputs = model(image)
                    pred = outputs.logits  # logits
                else:
                    pred = model(image)

        _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy()                       
                        
        TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1),label.reshape(-1),args.num_classes)
        TP_all += TP
        FP_all += FP
        TN_all += TN
        FN_all += FN
        n_valid_sample_all += n_valid_sample

    OA = np.sum(TP_all)*1.0 / n_valid_sample_all
    for i in range(args.num_classes):
        P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
        R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
        F1[i] = 2.0*P*R / (P + R + epsilon)
        IoU[i] = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + FN_all[i] + epsilon)
    
        if i==1:
            print('===>' + name_classes[i] + ' Precision: %.2f'%(P * 100))
            print('===>' + name_classes[i] + ' Recall: %.2f'%(R * 100))            
            print('===>' + name_classes[i] + ' IoU: %.2f'%(IoU[i] * 100))              
            print('===>' + name_classes[i] + ' F1: %.2f'%(F1[i] * 100))   
            f.write('===>' + name_classes[i] + ' Precision: %.2f\n'%(P * 100))
            f.write('===>' + name_classes[i] + ' Recall: %.2f\n'%(R * 100))            
            f.write('===>' + name_classes[i] + ' IoU: %.2f\n'%(IoU[i] * 100))              
            f.write('===>' + name_classes[i] + ' F1: %.2f\n'%(F1[i] * 100))   
        
    mF1 = np.mean(F1)   
    mIoU = np.mean(IoU)           
    print('===> mIoU: %.2f mean F1: %.2f OA: %.2f'%(mIoU*100,mF1*100,OA*100))
    f.write('===> mIoU: %.2f mean F1: %.2f OA: %.2f\n'%(mIoU*100,mF1*100,OA*100))        
    f.close()
    saved_state_dict = torch.load(os.path.join(snapshot_dir, model_name))  
    np.savez(snapshot_dir+'Precision_'+str(int(P * 10000))+'Recall_'+str(int(R * 10000))+'F1_'+str(int(F1[1] * 10000))+'_hist.npz',hist=hist) 
    
if __name__ == '__main__':
    main()
