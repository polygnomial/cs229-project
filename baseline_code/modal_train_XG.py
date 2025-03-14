import modal
import argparse

app = modal.App("train-sen2fire-xgboost-model")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root")
)

def get_arguments(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train XGBoost baseline on Sen2Fire dataset with vegetation indices.")
    
    # Data and dataset configuration
    parser.add_argument("--data_dir", type=str, default='/data', #Note chaning the path to your local path
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
    
    return parser.parse_args(args=args)

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=10000,
    volumes={"/data": modal.Volume.from_name("dataset-volume")}
)
def train_model(*arglist):
    args = get_arguments(args=arglist)
    import XGboost_baseline as train_model_impl
    train_model_impl.main(args)

if __name__ == "__main__":
    with app.run():
        train_model.remote()
