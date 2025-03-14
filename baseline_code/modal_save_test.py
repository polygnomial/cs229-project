from pathlib import Path

volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_DIR = Path("/models")

@app.function(gpu="any", volumes={MODEL_DIR: volume})  # attach the Volume
def train_model(data, config):
    import run_training

    model = run_training(config, data)
    model.save(config, MODEL_DIR)