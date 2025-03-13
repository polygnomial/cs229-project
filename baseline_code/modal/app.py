import modal
import boto3
import os

# Define constants for Cloudflare R2
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_BUCKET_NAME = "sen2fire"
R2_ENDPOINT_URL = "https://ece4d1a81301edbdafdd974f3da16020.r2.cloudflarestorage.com"

# Define the Modal Stub
app = modal.App("train-model-r2")

# Define the container environment with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
)

@app.function(image=image, gpu="T4", secrets=[modal.Secret.from_name("Cloudflare_R2")])
def train_model():
    import torch
    # Connect to R2 and download dataset
    s3_client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
    )

    local_filename = "scene4/scene_4_patch_9_9.npz"
    s3_client.download_file(R2_BUCKET_NAME, "scene4/scene_4_patch_9_9.npz", local_filename)

    # Load dataset
    print(torch.cuda.is_available())

    

# Run the Modal function
if __name__ == "__main__":
    with app.run():
        train_model.remote()
