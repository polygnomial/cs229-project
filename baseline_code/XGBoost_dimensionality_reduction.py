import argparse
import numpy as np
import os
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.Sen2Fire_Dataset import Sen2FireDataSet
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.ndimage import uniform_filter

# Define classes and a small epsilon to avoid division by zero
name_classes = np.array(['non-fire', 'fire'], dtype=str)
epsilon = 1e-14

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Incrementally train XGBoost on the Sen2Fire dataset with enhanced spatial-spectral features and optional dimensionality reduction."
    )
    # Dataset configuration
    parser.add_argument("--data_dir", type=str, default='/Users/d5826/desktop/milestone/Sen2Fire/',   # Path
                        help="Path to the Sen2Fire dataset directory.")
    parser.add_argument("--train_list", type=str, default='./dataset/train.txt',
                        help="Path to the training list file.")
    parser.add_argument("--val_list", type=str, default='./dataset/val.txt',
                        help="Path to the validation list file.")
    parser.add_argument("--test_list", type=str, default='./dataset/test.txt',
                        help="Path to the test list file (not used in training).")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes (non-fire and fire).")
    parser.add_argument("--input_strategy", type=str, default='swir', choices=['swir'],
                        help="Input strategy to use. Only 'swir' is supported.")
    parser.add_argument("--sample_pixels", type=int, default=-1,
                        help="Number of random pixels per patch to use for training. Set to -1 to load all pixels.")
    # XGBoost hyperparameters
    parser.add_argument("--max_depth", type=int, default=8, help="Maximum tree depth for XGBoost.")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate for XGBoost.")
    # Dimensionality reduction parameters
    parser.add_argument("--dim_reduction_method", type=str, default="autoencoder", choices=["none", "pca", "autoencoder"],
                        help="Dimensionality reduction method to apply on features.")
    parser.add_argument("--latent_dim", type=int, default=4,
                        help="Dimension of the latent space for PCA or Autoencoder.")
    parser.add_argument("--ae_epochs", type=int, default=50,
                        help="Number of epochs for Autoencoder pretraining.")
    parser.add_argument("--ae_lr", type=float, default=1e-3,
                        help="Learning rate for Autoencoder pretraining.")
    # Incremental training parameters
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of epochs to iterate over training patches.")
    parser.add_argument("--chunk_size", type=int, default=95,
                        help="Number of patches per training chunk.")
    parser.add_argument("--chunk_boost_rounds", type=int, default=10,
                        help="Number of boosting rounds for each training chunk.")
    # Prevent out-of-memory errors by automatically freeing GPU memory on exceptions
    parser.add_argument("--use_memory_guard", action="store_true",
                        help="Enable memory guard mechanism to release GPU cache on errors.")
    # Model saving and output paths
    parser.add_argument("--model_save_path", type=str, default='./xgboost_model.json',
                        help="Path to save the trained XGBoost model.")
    parser.add_argument("--snapshot_dir", type=str, default='./Map/',
                        help="Directory to save detection results and plots.")
    
    return parser.parse_args()

def extract_features(image, label, window_size=3):
    """
    According to the paper, the four-band approach (SWIR, NIR , Red , Aerosol) yields the best results. Therefore, use these four bands as the base.
    I also add additional features to improve, but we may drop them.

    Extract extended features, including:
      - Original bands: SWIR (B12, index 11), NIR (B8, index 7), Red (B4, index 3), Aerosol (B13, index 12)
      - Spectral indices:                                                                                          
            NBR = (NIR - SWIR) / (NIR + SWIR + epsilon)
            NDVI = (NIR - Red) / (NIR + Red + epsilon)
      - Local means (with window size=3) for each of the above 6 channels to incorporate spatial context
    After concatenation, each pixel ends up with a 12-dimensional feature (6 original + 6 context).
    """
    # Extract original channels
    swir = image[11:12, :, :]    # SWIR (B12)
    nir = image[7:8, :, :]       # NIR (B8)
    red = image[3:4, :, :]       # Red (B4)
    aerosol = image[12:13, :, :] # Aerosol (B13)
    # Calculate spectral indices
    nbr = (nir - swir) / (nir + swir + epsilon)
    ndvi = (nir - red) / (nir + red + epsilon)
    # Compute local mean
    def local_context(channel):
        return uniform_filter(channel, size=window_size, mode='reflect')
    swir_ctx = local_context(swir[0])
    nir_ctx = local_context(nir[0])
    red_ctx = local_context(red[0])
    aerosol_ctx = local_context(aerosol[0])
    nbr_ctx = local_context(nbr[0])
    ndvi_ctx = local_context(ndvi[0])
    # Concatenate original channels and spectral indices
    features = np.concatenate([swir, nir, red, aerosol, nbr, ndvi], axis=0)  # shape: (6, H, W)
    context = np.stack([swir_ctx, nir_ctx, red_ctx, aerosol_ctx, nbr_ctx, ndvi_ctx], axis=0)  # shape: (6, H, W)
    combined_features = np.concatenate([features, context], axis=0)  # shape: (12, H, W)
    X = combined_features.reshape(combined_features.shape[0], -1).T  # (H*W, 12)
    y = label.reshape(-1)
    return X, y

# Autoencoder for dimensionality reduction
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon

def gpu_pca_transform(X, latent_dim):
    # Convert X to tensor and perform GPU-based PCA
    X_tensor = torch.from_numpy(X).float().cuda()
    U, S, V = torch.pca_lowrank(X_tensor, q=latent_dim)
    X_reduced = torch.matmul(X_tensor, V[:, :latent_dim])
    return X_reduced.cpu().numpy()

def pretrain_autoencoder(train_loader, scaler, input_dim, latent_dim, ae_epochs, ae_lr):
    """
    Pretrain the Autoencoder module:
      - train_loader: training data loader, one patch at a time
      - scaler: fitted StandardScaler for normalization
      - input_dim: input feature dimension (12 in this example)
      - latent_dim: target dimension after reduction
      - ae_epochs: number of epochs
      - ae_lr: learning rate
    """
    autoencoder = Autoencoder(input_dim, latent_dim).cuda()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=ae_lr)
    criterion = nn.MSELoss()
    autoencoder.train()
    for epoch in range(ae_epochs):
        epoch_loss = 0.0
        count = 0
        for batch in tqdm(train_loader, desc=f"Autoencoder Pretraining Epoch {epoch+1}/{ae_epochs}"):
            image, label, _, _ = batch
            image = image.squeeze(0).numpy()
            if image.shape[0] < 13:
                continue
            X, _ = extract_features(image, label)
            X = scaler.transform(X)
            X_tensor = torch.from_numpy(X).float().cuda()
            optimizer.zero_grad()
            outputs = autoencoder(X_tensor)
            loss = criterion(outputs, X_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            count += 1
        if count > 0:
            print(f"Epoch {epoch+1}/{ae_epochs} Loss: {epoch_loss/count:.4f}")
    autoencoder.eval()
    return autoencoder

def eval_image(y_true, y_pred, n_class):
    TP = np.zeros((n_class, 1))
    FP = np.zeros((n_class, 1))
    TN = np.zeros((n_class, 1))
    FN = np.zeros((n_class, 1))
    for i in range(n_class):
        TP[i] = np.sum((y_true == i) & (y_pred == i))
        FP[i] = np.sum((y_true != i) & (y_pred == i))
        FN[i] = np.sum((y_true == i) & (y_pred != i))
        TN[i] = np.sum((y_true != i) & (y_pred != i))
    n_valid_sample = len(y_true)
    return TP, FP, TN, FN, n_valid_sample

def main(args):
    
    os.makedirs(args.snapshot_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    print("Loading datasets...")
    train_dataset = Sen2FireDataSet(args.data_dir, args.train_list, mode=None)
    val_dataset = Sen2FireDataSet(args.data_dir, args.val_list, mode=None)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=os.cpu_count(), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=os.cpu_count(), pin_memory=True)
    
    # Count the total number of positive and negative samples for imbalance handling
    total_pos, total_neg = 0, 0
    for batch in tqdm(train_loader, desc="Counting training samples"):
        image, label, _, _ = batch
        image = image.squeeze(0).numpy()
        label = label.squeeze(0).numpy()
        if image.shape[0] < 13:
            continue
        total_pos += np.sum(label == 1)
        total_neg += np.sum(label == 0)
    scale_pos_weight = total_neg / total_pos if total_pos > 0 else 1.0
    print(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Reset train_loader (since it was fully iterated once)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=os.cpu_count(), pin_memory=True)
    
    # Use partial_fit to fit StandardScaler patch by patch
    scaler = StandardScaler()
    for batch in tqdm(train_loader, desc="Fitting scaler"):
        image, label, _, _ = batch
        image = image.squeeze(0).numpy()
        label = label.squeeze(0).numpy()
        if image.shape[0] < 13:
            continue
        X, _ = extract_features(image, label)
        scaler.partial_fit(X)
    
    # Process validation data (load all validation data into memory)
    X_val_list, y_val_list = [], []
    for batch in tqdm(val_loader, desc="Processing Validation Patches"):
        image, label, _, _ = batch
        image = image.squeeze(0).numpy()
        label = label.squeeze(0).numpy()
        if image.shape[0] < 13:
            continue
        X, y = extract_features(image, label)
        X_val_list.append(X)
        y_val_list.append(y)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    X_val = scaler.transform(X_val).astype(np.float32)
    
    # Handle validation data dimensionality based on selected reduction method
    autoencoder = None
    if args.dim_reduction_method == "pca":
        print("Applying GPU-based PCA on validation data to reduce dimension to", args.latent_dim)
        X_val = gpu_pca_transform(X_val, args.latent_dim)
    elif args.dim_reduction_method == "autoencoder":
        print("Pretraining Autoencoder for dimensionality reduction...")
        autoencoder = pretrain_autoencoder(train_loader, scaler, input_dim=12, latent_dim=args.latent_dim,
                                           ae_epochs=args.ae_epochs, ae_lr=args.ae_lr)
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_val).float().cuda()
            X_val = autoencoder.encoder(X_tensor).cpu().numpy()
    X_val = X_val.astype(np.float32)
    
    # XGBoost parameter configuration
    params = {
        'max_depth': args.max_depth,
        'eta': args.learning_rate,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,  
        'tree_method': 'hist',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Incremental training: train model by chunk to prevent out-of-memory
    num_epochs = args.num_epochs
    chunk_size = args.chunk_size
    booster = None
    global_iter = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        chunk_X_list, chunk_y_list = [], []
        chunk_count = 0
        for batch in tqdm(train_loader, desc="Training epoch"):
            image, label, _, _ = batch
            image = image.squeeze(0).numpy()
            label = label.squeeze(0).numpy()
            if image.shape[0] < 13:
                continue
            X, y = extract_features(image, label)
            X = scaler.transform(X)
            if args.dim_reduction_method == "pca":
                X = gpu_pca_transform(X, args.latent_dim)
            elif args.dim_reduction_method == "autoencoder" and autoencoder is not None:
                with torch.no_grad():
                    X_tensor = torch.from_numpy(X).float().cuda()
                    X = autoencoder.encoder(X_tensor).cpu().numpy()
            chunk_X_list.append(X)
            chunk_y_list.append(y)
            chunk_count += 1
            if chunk_count >= chunk_size:
                X_chunk = np.concatenate(chunk_X_list, axis=0).astype(np.float32)
                y_chunk = np.concatenate(chunk_y_list, axis=0)
                dchunk = xgb.DMatrix(X_chunk, label=y_chunk)
                try:
                    if booster is None:
                        booster = xgb.train(params, dchunk, num_boost_round=args.chunk_boost_rounds)
                    else:
                        booster = xgb.train(params, dchunk, num_boost_round=args.chunk_boost_rounds, xgb_model=booster)
                except Exception as e:
                    print("Encountered error during training chunk:", e)
                    if args.use_memory_guard:
                        print("Releasing GPU cache...")
                        torch.cuda.empty_cache()
                    continue
                global_iter += 1
                chunk_X_list, chunk_y_list = [], []
                chunk_count = 0
                if args.use_memory_guard:
                    torch.cuda.empty_cache()
        if chunk_count > 0:
            X_chunk = np.concatenate(chunk_X_list, axis=0).astype(np.float32)
            y_chunk = np.concatenate(chunk_y_list, axis=0)
            dchunk = xgb.DMatrix(X_chunk, label=y_chunk)
            try:
                if booster is None:
                    booster = xgb.train(params, dchunk, num_boost_round=args.chunk_boost_rounds)
                else:
                    booster = xgb.train(params, dchunk, num_boost_round=args.chunk_boost_rounds, xgb_model=booster)
            except Exception as e:
                print("Encountered error during training chunk:", e)
                if args.use_memory_guard:
                    print("Releasing GPU cache...")
                    torch.cuda.empty_cache()
            global_iter += 1
        
        # Evaluate on the validation set after each epoch
        dval = xgb.DMatrix(X_val, label=y_val)
        y_pred_prob = booster.predict(dval)
        y_pred = (y_pred_prob > 0.5).astype(np.uint8)
        val_acc = accuracy_score(y_val, y_pred)
        TP, FP, TN, FN, n_valid_sample = eval_image(y_val, y_pred, args.num_classes)
        OA = np.sum(TP) / n_valid_sample
        P_fire = float(TP[1] / (TP[1] + FP[1] + epsilon))
        R_fire = float(TP[1] / (TP[1] + FN[1] + epsilon))
        F1_fire = float(2 * P_fire * R_fire / (P_fire + R_fire + epsilon))
        IoU_fire = float(TP[1] / (TP[1] + FP[1] + FN[1] + epsilon))
        F1_all = np.zeros((args.num_classes, 1))
        IoU_all = np.zeros((args.num_classes, 1))
        for i in range(args.num_classes):
            P = TP[i] / (TP[i] + FP[i] + epsilon)
            R = TP[i] / (TP[i] + FN[i] + epsilon)
            F1_all[i] = 2 * P * R / (P + R + epsilon)
            IoU_all[i] = TP[i] / (TP[i] + FP[i] + FN[i] + epsilon)
        mF1 = np.mean(F1_all)
        mIoU = np.mean(IoU_all)
        print(f"Epoch {epoch+1} Validation Accuracy: {val_acc:.4f}")
        print(f"===> fire Precision: {P_fire*100:.2f}")
        print(f"===> fire Recall: {R_fire*100:.2f}")
        print(f"===> fire IoU: {IoU_fire*100:.2f}")
        print(f"===> fire F1: {F1_fire*100:.2f}")
        print(f"===> mIoU: {mIoU*100:.2f} mean F1: {mF1*100:.2f} OA: {OA*100:.2f}")
    
    booster.save_model(args.model_save_path)
    print(f"Model saved to {args.model_save_path}")
    
    # Output final metrics on the validation set
    dval = xgb.DMatrix(X_val, label=y_val)
    y_pred_prob = booster.predict(dval)
    y_pred = (y_pred_prob > 0.5).astype(np.uint8)
    final_acc = accuracy_score(y_val, y_pred)
    TP, FP, TN, FN, n_valid_sample = eval_image(y_val, y_pred, args.num_classes)
    OA = np.sum(TP) / n_valid_sample
    P_fire = float(TP[1] / (TP[1] + FP[1] + epsilon))
    R_fire = float(TP[1] / (TP[1] + FN[1] + epsilon))
    F1_fire = float(2 * P_fire * R_fire / (P_fire + R_fire + epsilon))
    IoU_fire = float(TP[1] / (TP[1] + FP[1] + FN[1] + epsilon))
    F1_all = np.zeros((args.num_classes, 1))
    IoU_all = np.zeros((args.num_classes, 1))
    for i in range(args.num_classes):
        P = TP[i] / (TP[i] + FP[i] + epsilon)
        R = TP[i] / (TP[i] + FN[i] + epsilon)
        F1_all[i] = 2 * P * R / (P + R + epsilon)
        IoU_all[i] = TP[i] / (TP[i] + FP[i] + FN[i] + epsilon)
    mF1 = np.mean(F1_all)
    mIoU = np.mean(IoU_all)
    print(f"Final Validation Accuracy: {final_acc:.4f}")
    print(f"===> fire Precision: {P_fire*100:.2f}")
    print(f"===> fire Recall: {R_fire*100:.2f}")
    print(f"===> fire IoU: {IoU_fire*100:.2f}")
    print(f"===> fire F1: {F1_fire*100:.2f}")
    print(f"===> mIoU: {mIoU*100:.2f} mean F1: {mF1*100:.2f} OA: {OA*100:.2f}")

if __name__ == '__main__':
    main()
