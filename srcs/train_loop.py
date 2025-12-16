import torch
import tqdm
from srcs.Train.testing import testing
import os
import torch
import tqdm
from srcs.Train.numbers import target_accuracy

def train(model, train_loader, eval_loader, device, epochs=10, lr=1e-3,
          save_every=2, checkpoint_dir="./checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["y"].to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        avg_train_loss = running_loss / max(1, total_samples)
        print(f"Epoch {epoch+1} train loss: {avg_train_loss:.4f}")
        acc = testing(model, train_loader, eval_loader, device)
        print(f"Epoch {epoch+1} metrics -> acc: {acc:.4f}")

        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pt")
            print("ABOUT TO SAVE:", ckpt_path)
            print("  model type:", type(model))
            print("  has .to ?:", hasattr(model, "to"))
            torch.save(model, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        if acc > target_accuracy:
            ckpt_path = os.path.join(checkpoint_dir, "model_best.pt")
            print("ABOUT TO SAVE:", ckpt_path)
            print("  model type:", type(model))
            print("  has .to ?:", hasattr(model, "to"))
            torch.save(model, ckpt_path)
            print(f"Early stop at epoch {epoch+1}, saved best model: {ckpt_path}")
            break
