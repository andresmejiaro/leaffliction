import torch
import tqdm
from srcs.Train.testing import testing

def train(model, train_loader, eval_loader, device, epochs=10, lr=1e-3, save_every=2, checkpoint_dir="./checkpoints"):
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["y"].to(device)
            opt.zero_grad(set_to_none=True)
            opt.step()

        if (epoch+1) % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        if testing(model, train_loader, eval_loader, device) > 0.95:
            torch.save(model.state_dict(), ckpt_path)
            break
        
        print(f"Epoch {epoch+1} done.")