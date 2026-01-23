#%%
import random
import torch
import plotly.express as px
import numpy as np
from PIL import Image
from typing import Any
from torch.utils.data import DataLoader
from train import CSVDataset

def visualize_random_sample(model: torch.nn.Module, dataloader: Any, device: str = "cpu") -> None:
    torch_device = torch.device(device)
    model.to(torch_device).eval()

    batch = next(iter(dataloader))
    dataset = dataloader.dataset
    idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

    x = batch["image"].to(torch_device)     # [B,3,H,W]
    y = batch["y"]                    # [B]
    labels = batch["label"]           # list[str]
    paths = batch["path"]             # list[str]

    idx = random.randrange(x.size(0))

    # --- prediction ---
    with torch.no_grad():
        logits = model(x[idx:idx+1])
        pred_idx = torch.argmax(logits, dim=1).item()

    true_idx = y[idx].item()
    true_label = labels[idx]

    title = f"True: {true_label} | Pred: {idx_to_class[pred_idx]}"
    # --- load raw image from disk ---
    img = Image.open(paths[idx]).convert("RGB")
    img_np = np.array(img)

    fig = px.imshow(img_np, title=title)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()
#%%

model = torch.load("checkpoints/model_epoch2.pt", map_location="cpu", weights_only=False)


evaluation_data = CSVDataset("eval", "dataset.csv", root=".")

use_pin = torch.cuda.is_available()

evaluation_loader = DataLoader(evaluation_data,
                                   batch_size=16,
                                   shuffle=True,
                                   num_workers=2,
                                   pin_memory=use_pin)

CLASS_NAMES = ["1","2","3","4","5","6","8","9"]

# %%


visualize_random_sample(model,evaluation_loader)
# %%
