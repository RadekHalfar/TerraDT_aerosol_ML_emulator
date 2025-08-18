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
from sklearn.model_selection import TimeSeriesSplit
from copy import deepcopy
from SequenceKappaDataset import SequenceKappaDataset
from torch.amp import autocast, GradScaler
import platform

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
               plot=True, experiment_name="Default",
               n_splits: int = 5, gap: int = 0,
               show_fold_plot: bool = False,
               seq_len: int = 1):
    
    # Set up MLflow
    # Ensure MLflow writes to a path valid in WSL (POSIX). This avoids paths like '/C:/...'
    # that can appear when Windows-style paths are auto-detected and prefixed with '/'.
    tracking_dir = Path("mlruns").resolve()
    # Cross-platform: build a proper file URI (file:///...) that works on Windows and WSL
    desired_tracking_uri = tracking_dir.as_uri()
    # Strong override in case environment defaults are set
    os.environ["MLFLOW_TRACKING_URI"] = desired_tracking_uri
    mlflow.set_tracking_uri(desired_tracking_uri)

    # Derive environment-specific experiment name to avoid artifact path clashes
    def _is_wsl() -> bool:
        try:
            with open('/proc/version', 'r') as f:
                v = f.read().lower()
                return 'microsoft' in v or 'wsl' in v
        except Exception:
            return False

    if _is_wsl():
        env_suffix = 'wsl'
    elif platform.system().lower() == 'windows':
        env_suffix = 'win'
    else:
        env_suffix = 'posix'

    experiment_actual_name = f"{experiment_name}_{env_suffix}"

    # Resolve or create experiment with the artifact location under this environment
    exp = mlflow.get_experiment_by_name(experiment_actual_name)
    exp_id = None
    if exp is None:
        # New experiment pointing to the desired POSIX artifact location
        exp_id = mlflow.create_experiment(
            experiment_actual_name,
            artifact_location=desired_tracking_uri
        )
    else:
        # If existing experiment has a different artifact location, recreate with this env name
        current_loc = (exp.artifact_location or "")
        if not current_loc.startswith(desired_tracking_uri):
            exp_id = mlflow.create_experiment(
                experiment_actual_name,
                artifact_location=desired_tracking_uri
            )
        else:
            exp_id = exp.experiment_id

    # End any active run (from previous errors) before starting a new one
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:
        pass

    # Start run explicitly under the selected experiment id
    mlflow.start_run(experiment_id=exp_id)
    # Debug: confirm URIs used
    try:
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"MLflow artifact URI: {mlflow.get_artifact_uri()}")
        exp_meta = mlflow.get_experiment(exp_id)
        print(f"MLflow experiment: id={exp_meta.experiment_id}, name={exp_meta.name}, artifact_location={exp_meta.artifact_location}")
    except Exception:
        pass

    # Performance / stability settings for CUDA
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    if device == 'cuda':
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    
    # Log parameters
    mlflow.log_params({
        'batch_size': batch_size,
        'learning_rate': lr,
        'epochs': epochs,
        'device': device,
        'model_type': model.__class__.__name__,
        'seq_len': int(seq_len),
    })
    
    # Log model architecture
    mlflow.log_text(str(model), 'model_architecture.txt')

    # Chronological split indices using TimeSeriesSplit (no shuffling)
    full_dataset = nc.Dataset(nc_file_path)
    total_timesteps = len(full_dataset.dimensions['time'])
    full_dataset.close()

    indices = np.arange(total_timesteps)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    #val_dataset = KappaDataset(nc_file_path, indices=val_idx)
    #test_dataset = KappaDataset(nc_file_path, indices=test_idx)

    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if resume_from:
        base_results_dir = os.path.dirname(resume_from)
        print(f"📂 Resuming training, saving outputs to existing folder: {base_results_dir}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        base_results_dir = os.path.join("results", f"res_{timestamp}")
        os.makedirs(base_results_dir, exist_ok=True)

    
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        model, optimizer, start_epoch = load_checkpoint(resume_from, model, optimizer)

    # Save initial weights to reset for each fold
    initial_state = deepcopy(model.state_dict()) if model is not None else None

    all_fold_metrics = []
    last_fold_outputs = None

    # Iterate chronological folds
    for fold, (train_ind, val_ind) in enumerate(tscv.split(indices)):
        # Reset model to initial weights per fold
        if initial_state is not None:
            model.load_state_dict(initial_state)

        # Datasets and loaders (no shuffle to preserve time order)
        if int(seq_len) > 1:
            Xtrain = SequenceKappaDataset(nc_file_path, indices=train_ind, seq_len=int(seq_len))
            Xval = SequenceKappaDataset(nc_file_path, indices=val_ind, seq_len=int(seq_len))
        else:
            Xtrain = KappaDataset(nc_file_path, indices=train_ind)
            Xval = KappaDataset(nc_file_path, indices=val_ind)

        # DataLoader performance settings
        use_cuda = str(device).startswith('cuda')
        workers = max(1, min(8, (os.cpu_count() or 2) // 2))
        common_kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "pin_memory": use_cuda,
            "num_workers": workers,
            "persistent_workers": workers > 0,
            "drop_last": True,
        }
        # prefetch_factor is only valid if num_workers > 0
        if workers > 0:
            common_kwargs["prefetch_factor"] = 2

        train_loader = DataLoader(Xtrain, **common_kwargs)
        val_loader = DataLoader(Xval, **common_kwargs)

        # Per-fold results directory
        results_dir = os.path.join(base_results_dir, f"fold_{fold}")
        os.makedirs(results_dir, exist_ok=True)
        log_path = os.path.join(results_dir, "val_loss_log.txt")
        training_plot_path = os.path.join(results_dir, "training_curve.png")

        train_losses, val_losses = [], []
        best_val_loss = float('inf')

        # AMP scaler for mixed precision (enabled only on CUDA)
        #scaler = GradScaler(enabled=(device == 'cuda'))
        scaler = GradScaler(enabled=False)

        for epoch in range(start_epoch, epochs):
            model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f"Fold {fold} | Epoch {epoch+1}/{epochs} [Training]", leave=False)

            for batch_idx, (x_batch, y_batch) in enumerate(pbar):
                # non_blocking copies work with pinned memory and CUDA
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                optimizer.zero_grad()
                try:
                    #with autocast(device_type='cuda', enabled=(device == 'cuda')):
                    with autocast(device_type='cuda', enabled=False):
                        preds = model(x_batch)
                        loss = loss_fn(preds, y_batch)
                except RuntimeError as e:
                    if 'unable to find an engine' in str(e).lower():
                        # Fallback to full precision for this batch
                        preds = model(x_batch)
                        loss = loss_fn(preds, y_batch)
                    else:
                        raise
                # Backward with AMP
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                current_loss = loss.item()
                pbar.set_postfix({"Loss": f"{current_loss:.4f}"})

                # Log batch metrics (less frequently)
                if batch_idx % 50 == 0:  # Log every 50 batches
                    mlflow.log_metric(f'fold{fold}/train/batch_loss', current_loss,
                                      step=epoch * len(train_loader) + batch_idx)
                    log_system_metrics(epoch * len(train_loader) + batch_idx)

            train_loss /= max(1, len(train_loader))
            train_losses.append(train_loss)
            mlflow.log_metric(f'fold{fold}/train/epoch_loss', train_loss, step=epoch)

            # Validation
            model.eval()
            val_loss = 0
            pbar_val = tqdm(val_loader, desc=f"Fold {fold} | Epoch {epoch+1}/{epochs} [Validation]", leave=False)
            with torch.no_grad():
                for x_val, y_val in pbar_val:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    try:
                        #with autocast(device_type='cuda', enabled=(device == 'cuda')):
                        with autocast(device_type='cuda', enabled=False):
                            preds = model(x_val)
                            loss = loss_fn(preds, y_val)
                    except RuntimeError as e:
                        if 'unable to find an engine' in str(e).lower():
                            preds = model(x_val)
                            loss = loss_fn(preds, y_val)
                        else:
                            raise
                    val_loss += loss.item()
                    pbar_val.set_postfix({"ValLoss": f"{loss.item():.4f}"})

            val_loss /= max(1, len(val_loader))
            val_losses.append(val_loss)
            mlflow.log_metric(f'fold{fold}/val/epoch_loss', val_loss, step=epoch)

            print(f"✅ Fold {fold} Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # Log best model within the run
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                #mlflow.pytorch.log_model(model, f"fold{fold}/best_model")
                mlflow.pytorch.log_model(model, name=f"fold{fold}_best_model")

                with open(log_path, "a") as f:
                    f.write(f"[BEST] Epoch {epoch+1}: Val Loss = {val_loss:.6f}, Train Loss = {train_loss:.6f}\n")
                print(f"📦 Fold {fold} best model updated at epoch {epoch+1}")
            else:
                with open(log_path, "a") as f:
                    f.write(f"Epoch {epoch+1}: Val Loss = {val_loss:.6f}, Train Loss = {train_loss:.6f}\n")

        # Plot and save learning curves for the fold
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
            plt.title(f"Fold {fold} - Training and Validation Loss")
            plt.savefig(training_plot_path)

            # Log the plot to MLflow
            mlflow.log_artifact(training_plot_path)

            # Show or close to avoid blocking
            if show_fold_plot:
                plt.show()
            else:
                plt.close(fig)

        # Log the fold directory with all artifacts
        mlflow.log_artifacts(results_dir, artifact_path=f"training_artifacts/fold_{fold}")

        all_fold_metrics.append({
            'fold': fold,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
        })

        last_fold_outputs = (model, train_losses, val_losses)

    print("✅ TimeSeriesSplit training complete.")
    
    # Log final metrics and artifacts
    # Log summary metrics across folds (best of each)
    for m in all_fold_metrics:
        mlflow.log_metric(f"fold{m['fold']}/best_val_loss", m['best_val_loss'])
        if m['final_train_loss'] is not None:
            mlflow.log_metric(f"fold{m['fold']}/final_train_loss", m['final_train_loss'])
        if m['final_val_loss'] is not None:
            mlflow.log_metric(f"fold{m['fold']}/final_val_loss", m['final_val_loss'])
    
    # Log the code as an artifact
    mlflow.log_artifact(__file__)
    
    # End the MLflow run
    mlflow.end_run()
    
    # Return the outputs from the last fold
    return last_fold_outputs if last_fold_outputs is not None else (model, [], [])
