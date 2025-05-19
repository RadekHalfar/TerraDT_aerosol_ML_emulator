import netCDF4 as nc
import numpy as np
import torch
import matplotlib.pyplot as plt
from KappaDataset import KappaDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from datetime import datetime
import torch.nn as nn
from utils import load_checkpoint

def train_model(nc_file_path, resume_from=None, epochs=50, batch_size=4, lr=1e-3, model=None, optimizer=None, loss_fn=None, device = 'cuda' if torch.cuda.is_available() else 'cpu', plot=True):
    # Split indices
    full_dataset = nc.Dataset(nc_file_path)
    total_timesteps = len(full_dataset.dimensions['time'])
    full_dataset.close()

    indices = np.arange(total_timesteps)
    np.random.seed(42)
    np.random.shuffle(indices)

    train_split = int(0.8 * total_timesteps)
    val_split = int(0.9 * total_timesteps)

    train_idx = indices[:train_split]
    val_idx = indices[train_split:val_split]
    test_idx = indices[val_split:]

    train_dataset = KappaDataset(nc_file_path, indices=train_idx)
    val_dataset = KappaDataset(nc_file_path, indices=val_idx)
    test_dataset = KappaDataset(nc_file_path, indices=test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if resume_from:
        results_dir = os.path.dirname(resume_from)
        print(f"📂 Resuming training, saving outputs to existing folder: {results_dir}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        results_dir = os.path.join("results", f"res_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "val_loss_log.txt")
    best_model_path = os.path.join(results_dir, "best_model_checkpoint.pth")
    latest_model_path = os.path.join(results_dir, "latest_model_checkpoint.pth")
    training_plot_path = os.path.join(results_dir, "training_curve.png")


    start_epoch = 0  # Default if not resuming

    if resume_from and os.path.exists(resume_from):
        model, optimizer, start_epoch = load_checkpoint(resume_from, model, optimizer)


    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)

        for x_batch, y_batch in pbar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False)
        with torch.no_grad():
            for x_val, y_val in pbar_val:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds = model(x_val)
                loss = loss_fn(preds, y_val)
                val_loss += loss.item()
                pbar_val.set_postfix({"ValLoss": f"{loss.item():.4f}"})

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"✅ Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save latest model each epoch
        latest_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
        }
        torch.save(latest_checkpoint, latest_model_path)

        # Save best model if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(latest_checkpoint, best_model_path)

            with open(log_path, "a") as f:
                f.write(f"[BEST] Epoch {epoch+1}: Val Loss = {val_loss:.6f}, Train Loss = {train_loss:.6f}\n")

            print(f"📦 Best model updated at epoch {epoch+1}, saved to {best_model_path}")
        else:
            with open(log_path, "a") as f:
                f.write(f"Epoch {epoch+1}: Val Loss = {val_loss:.6f}, Train Loss = {train_loss:.6f}\n")

    # Plot and save learning curves
    if plot:
        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.plot(train_losses, color='blue', label='Train Loss')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Loss (MSE)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(val_losses, color='red', label='Validation Loss')
        ax2.set_ylabel("Validation Loss (MSE)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        fig.tight_layout()

        plt.savefig(training_plot_path)
        plt.show()

    print(f"📊 Training curve saved to {training_plot_path}")
    print("✅ Training complete.")
