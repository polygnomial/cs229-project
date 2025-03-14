import modal
import argparse

app = modal.App("train-sen2fire-xgboost-dimensionality-reduction-model")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root")
)

def get_arguments(args=None):
    
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
    return parser.parse_args(args=args)

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=10000,
    volumes={"/data": modal.Volume.from_name("dataset-volume")}
)
def train_model(*arglist):
    args = get_arguments(args=arglist)
    import XGBoost_dimensionality_reduction as train_model_impl
    train_model_impl.main(args)

if __name__ == "__main__":
    with app.run():
        train_model.remote()
