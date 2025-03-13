import modal

app = modal.App()
image = modal.Image.debian_slim().pip_install("torch")

@app.function(gpu="A100", image=image)
def run():
    import torch
    print(torch.cuda.is_available())
