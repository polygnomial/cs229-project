import modal
import argparse

app = modal.App("train-sen2fire-segformer")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root")
)

def get_arguments(args=None):
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
    
    return parser.parse_args(args=args)

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=10000,
    volumes={"/data": modal.Volume.from_name("dataset-volume")}
)
def train_model(*arglist):
    args = get_arguments(args=arglist)
    import Segform_2 as train_model_impl
    train_model_impl.main(args)

if __name__ == "__main__":
    with app.run():
        train_model.remote()
