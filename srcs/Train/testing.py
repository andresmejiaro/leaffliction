import torch
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm

def print_test_result(t_loss, e_loss, t_accuracy, e_accuracy, t_sens, e_sens, t_spec, e_spec):
    """
    Prints a nice table comparing train and eval metrics.
    """
    headers = ["Metric", "Train", "Eval"]
    table = [
        ["Loss", f"{t_loss:.4f}", f"{e_loss:.4f}"],
        ["Accuracy", f"{t_accuracy:.4f}", f"{e_accuracy:.4f}"],
        ["Sensitivity", f"{t_sens:.4f}", f"{e_sens:.4f}"],
        ["Specificity", f"{t_spec:.4f}", f"{e_spec:.4f}"],
    ]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def loss(model, dataloader, device="cpu"):
    """
    Compute average cross-entropy loss over a dataloader.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            #print("starting loss....")
            x = batch["image"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            #print("logits loaded...")
            batch_loss = criterion(logits, y)
            #print("criterion done...")
            total_loss += batch_loss.item() * x.size(0)
            #print("batch_loss.item passed....")
            total_samples += x.size(0)
            #print("batch ended, continuing...")

    return total_loss / total_samples if total_samples > 0 else float("nan")


def calculate_metrics(model, dataloader, device="cpu"):
    """
    Compute accuracy, sensitivity, and specificity for binary classification.
    Vectorized version (fast, no Python loops).
    Assumes labels: 0 = negative, 1 = positive.
    """
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device)
            y_true = batch["y"].to(device)

            logits = model(x)
            y_pred = torch.argmax(logits, dim=1)

            all_true.append(y_true)
            all_pred.append(y_pred)

    if not all_true:
        return 0.0, 0.0, 0.0

    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)

    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(1, total)
    sensitivity = tp / max(1, (tp + fn))   # recall
    specificity = tn / max(1, (tn + fp))

    return accuracy, sensitivity, specificity


def testing(model, train_loader, eval_loader, device="cpu"):
    """
    Evaluate the model on training and evaluation sets, print results.
    """
    print("Running tests...")

    # Compute loss
    t_loss = loss(model, train_loader, device)
    e_loss = loss(model, eval_loader, device)

    # Compute metrics
    t_accuracy, t_sens, t_spec = calculate_metrics(model, train_loader, device)
    e_accuracy, e_sens, e_spec = calculate_metrics(model, eval_loader, device)

    # Print results
    print_test_result(t_loss, e_loss, t_accuracy, e_accuracy, t_sens, e_sens, t_spec, e_spec)
    return e_accuracy
