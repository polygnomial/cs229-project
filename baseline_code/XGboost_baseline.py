import argparse
import numpy as np
import os
import xgboost as xgb
import torch
from torch.utils import data
from dataset.Sen2Fire_Dataset import Sen2FireDataSet
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from model.Autoencoder import Autoencoder
import torch
from torch import nn
import torchvision.transforms as transforms
from utils.tools import *

# Define class labels and a small constant to avoid division by zero
name_classes = np.array(['non-fire', 'fire'], dtype=str)
epsilon = 1e-14



def extract_features_labels(image, label):
    """Extract pixel-wise features and labels from a patch."""
    C, H, W = image.shape
    X = image.reshape(C, -1).T  # Shape: (H*W, C)
    y = label.reshape(-1)       
    return X, y

def compute_vegetation_indices(image):
    """Compute additional vegetation indices: NBR and NDVI."""
    NIR = image[7]    # Near-Infrared
    Red = image[3]    # Red band
    SWIR = image[11]  # Shortwave Infrared
    
    NBR = (NIR - SWIR) / (NIR + SWIR + epsilon)  # Normalized Burn Ratio
    NDVI = (NIR - Red) / (NIR + Red + epsilon)     # Normalized Difference Vegetation Index
    return np.stack([NBR, NDVI], axis=0)

def sample_pixels(X, y, num_samples):
    """Randomly sample a subset of pixels to reduce data size."""
    total_pixels = X.shape[0]
    if num_samples < total_pixels:
        indices = np.random.choice(total_pixels, num_samples, replace=False)
        return X[indices], y[indices]
    return X, y

def main(args):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Ensure output directory exists
    os.makedirs(args.snapshot_dir, exist_ok=True)

    f = open(args.snapshot_dir+'Training_log.txt', 'w')
    
    # Load training and validation datasets
    print("Loading datasets...")
    train_dataset = Sen2FireDataSet(args.data_dir, args.train_list, mode=args.mode)
    val_dataset = Sen2FireDataSet(args.data_dir, args.val_list, mode=args.mode)
    
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    
    # Extract training features
    X_train_list, y_train_list = [], []
    for batch in tqdm(train_loader, desc="Training Patches"):
        image, label, _, _ = batch  
        image = image.squeeze(0).numpy()  
        label = label.squeeze(0).numpy()    
        
        if image.shape[0] >= 12:
            extra_features = compute_vegetation_indices(image)
            image = np.vstack([image, extra_features])
        
        X, y = extract_features_labels(image, label)
        X_sample, y_sample = sample_pixels(X, y, args.sample_pixels)
        X_train_list.append(X_sample)
        y_train_list.append(y_sample)
    
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # PCA
    if args.format == 'PCA':
        N_DIM = 100 # tuning hyperparameter
        pca = PCA(n_components=N_DIM)
        X_train = pca.fit_transform(X_train)
    elif args.format == 'Autoencoder':
        N_DIM = 100

        # Train autoencoder
        autoencoder = Autoencoder(input_dim=X_train[1], output_dim=N_DIM).to(device)
        optimizer = torch.optim.Adam(autoencoder.parameters())
        criterion = nn.MSELoss()
        num_epochs = 100

        for epoch in range(num_epochs):
            output = autoencoder(X_train)
            loss = criterion(output, X_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        X_train = autoencoder.encode(X_train)
    elif args.format == 'CNN':
        ### two ideas: we could build a simple CNN ourselves,
        ### or use a pretrained model
        pass
    
    # Compute class imbalance weight
    num_fire = np.sum(y_train == 1)
    num_non_fire = np.sum(y_train == 0)
    scale_pos_weight = num_non_fire / max(num_fire, 1)
    
    params = {
        'max_depth': args.max_depth,
        'eta': args.learning_rate,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'hist',
        'device': 'cuda'
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    bst = xgb.train(params, dtrain, num_boost_round=args.n_estimators)
    bst.save_model(args.model_save_path)
    
    print(f"Model saved to {args.model_save_path}")
    
    # Evaluation
    X_val_list, y_val_list = [], []
    for batch in tqdm(val_loader, desc="Validation Patches"):
        image, label, _, _ = batch
        image = image.squeeze(0).numpy()
        label = label.squeeze(0).numpy()
        
        if image.shape[0] >= 12:
            extra_features = compute_vegetation_indices(image)
            image = np.vstack([image, extra_features])
        
        X, y = extract_features_labels(image, label)
        X_val_list.append(X)
        y_val_list.append(y)
    
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    X_val = scaler.transform(X_val)
    if args.format == 'PCA':
        X_val = pca.transform(X_val)
    if args.format == 'Autoencoder':
        X_val = autoencoder.encode(X_val)
    
    dval = xgb.DMatrix(X_val, label=y_val)
    y_pred_prob = bst.predict(dval)
    y_pred = (y_pred_prob > 0.5).astype(np.uint8)
    
    print('Testing..........')  
    f.write('Testing..........\n')  

    # ===== FIX: Remove the erroneous xgb.eval() call =====
    # xgb.eval()   <-- This line is removed

    TP_all = np.zeros((args.num_classes, 1))
    FP_all = np.zeros((args.num_classes, 1))
    TN_all = np.zeros((args.num_classes, 1))
    FN_all = np.zeros((args.num_classes, 1))
    n_valid_sample_all = 0
    F1 = np.zeros((args.num_classes, 1))
    IoU = np.zeros((args.num_classes, 1))

    test_loader = data.DataLoader(
                    Sen2FireDataSet(args.data_dir, args.test_list, mode=args.mode),
                    batch_size=1, shuffle=False, pin_memory=True)

    # ---- For XGBoost-based evaluation ----
    # Process each test patch the same way as in training/validation
    X_test_list, y_test_list = [], []
    for batch in tqdm(test_loader, desc="Test Patches"):
        image, label, _, _ = batch  
        image = image.squeeze(0).numpy()  
        label = label.squeeze(0).numpy()    
        if image.shape[0] >= 12:
            extra_features = compute_vegetation_indices(image)
            image = np.vstack([image, extra_features])
        X, y = extract_features_labels(image, label)
        X_test_list.append(X)
        y_test_list.append(y)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    # Apply the same normalization and feature transformation as in training
    X_test = scaler.transform(X_test)
    if args.format == 'PCA':
        X_test = pca.transform(X_test)
    elif args.format == 'Autoencoder':
        X_test = autoencoder.encode(X_test)
    
    dtest = xgb.DMatrix(X_test, label=y_test)
    y_pred_prob = bst.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(np.uint8)
    
    # Evaluate pixel-wise using the provided evaluation function
    TP, FP, TN, FN, n_valid_sample = eval_image(y_pred, y_test, args.num_classes)
    TP_all += TP
    FP_all += FP
    TN_all += TN
    FN_all += FN
    n_valid_sample_all += n_valid_sample



    OA = np.sum(TP_all)*1.0 / n_valid_sample_all
    for i in range(args.num_classes):
        P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
        R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
        F1[i] = 2.0 * P * R / (P + R + epsilon)
        IoU[i] = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + FN_all[i] + epsilon)
        
        if i == 1:
            print('===>' + name_classes[i] + ' Precision: %.2f' % (P * 100))
            print('===>' + name_classes[i] + ' Recall: %.2f' % (R * 100))            
            print('===>' + name_classes[i] + ' IoU: %.2f' % (IoU[i] * 100))              
            print('===>' + name_classes[i] + ' F1: %.2f' % (F1[i] * 100))
            f.write('===>' + name_classes[i] + ' Precision: %.2f\n' % (P * 100))
            f.write('===>' + name_classes[i] + ' Recall: %.2f\n' % (R * 100))
            f.write('===>' + name_classes[i] + ' IoU: %.2f\n' % (IoU[i] * 100))
            f.write('===>' + name_classes[i] + ' F1: %.2f\n' % (F1[i] * 100))
            
    mF1 = np.mean(F1)   
    mIoU = np.mean(IoU)           
    print('===> mIoU: %.2f mean F1: %.2f OA: %.2f' % (mIoU*100, mF1*100, OA*100))
    f.write('===> mIoU: %.2f mean F1: %.2f OA: %.2f\n' % (mIoU*100, mF1*100, OA*100))
    f.close()

    # ===== Optionally =====
    # If you need to load a saved state dict or save history, adjust the variables accordingly.
    # For example, if you have a model name and history (hist) defined:
    # saved_state_dict = torch.load(os.path.join(args.snapshot_dir, 'model_name.pth'))
    # np.savez(os.path.join(args.snapshot_dir, 'Precision_'+str(int(P * 10000))+'Recall_'+str(int(R * 10000))+
    #          'F1_'+str(int(F1[1] * 10000))+'_hist.npz'), hist=hist)
    
if __name__ == '__main__':
    main()
