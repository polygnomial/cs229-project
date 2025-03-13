import modal
import os
from pathlib import Path

# Define Modal App
app = modal.App("upload-local-dataset")

# Use or create the persistent Modal Volume
volume = modal.Volume.from_name("dataset-volume")

# Set local dataset path (update this to your actual dataset location)
LOCAL_DATASET_PATH = Path("/Users/flynn/Sen2Fire")  # <-- update if needed

# Batch size (number of files per upload)
BATCH_SIZE = 50  # Adjust as desired

@app.function(volumes={"/data": volume})
def upload_files_to_modal(file_data: dict):
    """
    This function runs in Modal and writes the provided file data to the mounted volume.
    The volume is mounted at /data, so we use standard Python file I/O.
    """
    for modal_path, file_content in file_data.items():
        # Construct the full path in the mounted volume
        full_path = os.path.join("/data", modal_path)
        # Ensure that the target directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        print(f"⬆️ Uploading {modal_path} to {full_path}...")
        # Write the file content (binary mode)
        with open(full_path, "wb") as f:
            f.write(file_content)
    print("✅ Batch uploaded successfully to Modal Volume!")

@app.local_entrypoint()
def upload_local_files():
    """
    This function runs locally: it reads files from your local disk in batches
    and calls the Modal function to write them into the volume.
    """
    if not LOCAL_DATASET_PATH.exists():
        raise ValueError(f"❌ Local dataset path not found: {LOCAL_DATASET_PATH}")
    print(f"📂 Preparing dataset from {LOCAL_DATASET_PATH} for upload to Modal Volume...")

    # Build a list of files while preserving folder structure
    file_paths = []
    for file_path in LOCAL_DATASET_PATH.rglob("*"):
        # Exclude hidden files (like .DS_Store)
        if file_path.is_file() and not file_path.name.startswith("."):
            modal_path = str(file_path.relative_to(LOCAL_DATASET_PATH))
            file_paths.append((modal_path, file_path))

    # Upload files in batches
    for i in range(0, len(file_paths), BATCH_SIZE):
        batch = file_paths[i : i + BATCH_SIZE]
        # Create a dictionary: modal_path -> file bytes
        file_data = {modal_path: local_path.read_bytes() for modal_path, local_path in batch}
        print(f"🚀 Uploading batch {i // BATCH_SIZE + 1} ({len(batch)} files)...")
        upload_files_to_modal.remote(file_data)

    print("✅ All files uploaded successfully to Modal Volume!")
