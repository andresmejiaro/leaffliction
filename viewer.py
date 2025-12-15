
# %%
import random
import torch
import plotly.express as px
from torch.utils.data import DataLoader
from srcs.Train.data_management import CSVDatasetF3

#%%
def visualize_random_sample(model, dataloader, device="cpu", class_names=None):
    """
    Pick a random image from a dataloader, run it through the model,
    and show it with true vs predicted label.
    
    Assumes each batch is a dict with:
        batch["image"] -> tensor [B, C, H, W]
        batch["y"]     -> tensor [B] with class indices
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    # --- get one random batch ---
    batch = next(iter(dataloader))
    x = batch["image"].to(device)
    y_true = batch["y"].to(device)

    # pick random index in this batch
    idx = random.randrange(x.size(0))

    # --- forward pass (no grad needed for viz) ---
    with torch.no_grad():
        logits = model(x[idx:idx+1])          # keep batch dim: [1, C, H, W]
        pred_class = torch.argmax(logits, dim=1).item()

    true_class = y_true[idx].item()

    # --- build label strings ---
    if class_names is not None:
        true_str = f"{true_class} ({class_names[true_class]})"
        pred_str = f"{pred_class} ({class_names[pred_class]})"
    else:
        true_str = str(true_class)
        pred_str = str(pred_class)

    title = f"Real: {true_str} | Predicted: {pred_str}"

    # --- convert image tensor -> numpy for plotly ---
    img = x[idx].detach().cpu()               # [C, H, W] on CPU

    # if it’s CHW, move to HWC
    if img.ndim == 3:
        img = img.permute(1, 2, 0)           # [H, W, C]

    img_np = img.numpy()

    # if grayscale single-channel, squeeze last dim
    if img_np.ndim == 3 and img_np.shape[2] == 1:
        img_np = img_np[:, :, 0]

    fig = px.imshow(img_np, title=title)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()

#%%

model = torch.load("model", map_location="cpu")


evaluation_data = CSVDatasetF3("eval", "dataset.csv", root=".")

use_pin = torch.cuda.is_available()

evaluation_loader = DataLoader(evaluation_data,
                                   batch_size=16,
                                   shuffle=True,
                                   num_workers=2,
                                   pin_memory=use_pin)

CLASS_NAMES = ["1","2","3","4","5","6","8","9"]

# %%


visualize_random_sample(model,evaluation_data)