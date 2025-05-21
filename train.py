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
import mlflow
import mlflow.pytorch
from pathlib import Path
import psutil
import GPUtil

def log_system_metrics(step):
    """Log system metrics"""
    # CPU metrics
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    # GPU metrics if available
    gpu_metrics = {}
    try:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            gpu_metrics[f'gpu_{i}_load'] = gpu.load * 100
            gpu_metrics[f'gpu_{i}_memory_util'] = gpu.memoryUtil * 100
    except:
        pass
    
    # Log all metrics
    mlflow.log_metrics({
        'system/cpu_percent': cpu_percent,
        'system/memory_percent': memory.percent,
        'system/memory_available_gb': memory.available / (1024**3),
        **gpu_metrics
    }, step=step)

def train_model(nc_file_path, resume_from=None, epochs=50, batch_size=4, lr=1e-3, 
               model=None, optimizer=None, loss_fn=None, 
               device='cuda' if torch.cuda.is_available() else 'cpu', 
               plot=True, experiment_name="Default"):
    
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    
    # Log parameters
    mlflow.log_params({
        'batch_size': batch_size,
        'learning_rate': lr,
        'epochs': epochs,
        'device': device,
        'model_type': model.__class__.__name__
    })
    
    # Log model architecture
    mlflow.log_text(str(model), 'model_architecture.txt')

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

    # Log artifacts directory
    mlflow.log_artifact(results_dir, "training_artifacts")

    
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        model, optimizer, start_epoch = load_checkpoint(resume_from, model, optimizer)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)

        for batch_idx, (x_batch, y_batch) in enumerate(pbar):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            current_loss = loss.item()
            pbar.set_postfix({"Loss": f"{current_loss:.4f}"})
            
            # Log batch metrics
            if batch_idx % 10 == 0:  # Log every 10 batches
                mlflow.log_metric('train/batch_loss', current_loss, step=epoch * len(train_loader) + batch_idx)
                log_system_metrics(epoch * len(train_loader) + batch_idx)

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        mlflow.log_metric('train/epoch_loss', train_loss, step=epoch)

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
        mlflow.log_metric('val/epoch_loss', val_loss, step=epoch)

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
        
        # Log model checkpoint as artifact
        mlflow.pytorch.log_model(model, f"models/epoch_{epoch+1}")

        # Save best model if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(latest_checkpoint, best_model_path)
            mlflow.log_artifact(best_model_path, "best_model")
            
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
        plt.title("Training and Validation Loss")
        plt.savefig(training_plot_path)

        # Log the plot to MLflow
        mlflow.log_artifact(training_plot_path)

        plt.show()
        #plt.close(fig)
        


    print(f"📊 Training curve saved to {training_plot_path}")
    print("✅ Training complete.")
    
    # Log final metrics and artifacts
    mlflow.log_metrics({
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    })
    
    # Log the code as an artifact
    mlflow.log_artifact(__file__)
    
    # End the MLflow run
    mlflow.end_run()
    
    return model, train_losses, val_losses
