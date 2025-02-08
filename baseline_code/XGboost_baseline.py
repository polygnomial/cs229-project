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

# Define class labels and a small constant to avoid division by zero
name_classes = np.array(['non-fire', 'fire'], dtype=str)
epsilon = 1e-14

def get_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train XGBoost baseline on Sen2Fire dataset with vegetation indices.")
    
    # Data and dataset configuration
    parser.add_argument("--data_dir", type=str, default='/Users/d5826/desktop/milestone/Sen2Fire/', #Note chaning the path to your local path
                        help="Path to the Sen2Fire dataset directory.")
    parser.add_argument("--train_list", type=str, default='./dataset/train.txt',
                        help="Path to the training list file.")
    parser.add_argument("--val_list", type=str, default='./dataset/val.txt',
                        help="Path to the validation list file.")
    parser.add_argument("--test_list", type=str, default='./dataset/test.txt',
                        help="Path to the test list file (not used in training).")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes (non-fire and fire).")
    parser.add_argument("--mode", type=int, default=10,
                        help="Input mode (e.g., 10 for 'rgb_swir_nbr_ndvi').")
    parser.add_argument("--format", type=str, default='xgboost_direct',
                        help="input transform type.")     
    
    # XGBoost hyperparameters (initial/default values)
    parser.add_argument("--max_depth", type=int, default=6, help="Maximum tree depth for XGBoost.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of boosting rounds (trees).")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for XGBoost.")
    
    # Pixel sampling configuration
    parser.add_argument("--sample_pixels", type=int, default=1000,
                        help="Number of random pixels per patch to use for training.")
    
    # Model saving and output paths
    parser.add_argument("--model_save_path", type=str, default='./xgboost_model.json',
                        help="Path to save the trained XGBoost model.")
    parser.add_argument("--snapshot_dir", type=str, default='./Map/',
                        help="Directory to save detection results and plots.")
    
    return parser.parse_args()

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

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = get_arguments()
    
    # Ensure output directory exists
    os.makedirs(args.snapshot_dir, exist_ok=True)
    
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
    
    print(f"Validation Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    
if __name__ == '__main__':
    main()
