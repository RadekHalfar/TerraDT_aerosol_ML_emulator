import torch
import matplotlib.pyplot as plt
from model_CNN_test import KappaPredictorCNN
from KappaDataset import KappaDataset
import mlflow
import mlflow.pytorch
import numpy as np
from typing import List, Optional
from pathlib import Path
from mlflow.tracking import MlflowClient

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    print(f"🔄 Loaded checkpoint from {checkpoint_path}, starting at epoch {start_epoch}")
    return model, optimizer, start_epoch

def visualize_prediction(model_path, nc_path, time_idx=0, lev_indices=[0], device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load dataset and model
    dataset = KappaDataset(nc_path)
    x, y_true = dataset[time_idx]
    x = x.unsqueeze(0).to(device)

    model = KappaPredictorCNN(in_channels=12).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Predict
    with torch.no_grad():
        y_pred = model(x).squeeze(0).cpu().numpy()  # [2, lev, lat, lon]
    y_true = y_true.numpy()

    num_levels = len(lev_indices)

    # Set up correct subplot shape
    fig_su, axes_su = plt.subplots(2, num_levels, figsize=(5 * num_levels, 8))

    # Ensure axes is always 2D array for consistent indexing
    if num_levels == 1:
        axes_su = np.expand_dims(axes_su, axis=1)

    # --- kappa_SU ---
    for i, lev in enumerate(lev_indices):
        im0 = axes_su[0, i].imshow(y_true[0, lev], cmap='viridis')
        axes_su[0, i].set_title(f"True kappa_SU (lev={lev})")
        plt.colorbar(im0, ax=axes_su[0, i])

        im1 = axes_su[1, i].imshow(y_pred[0, lev], cmap='viridis')
        axes_su[1, i].set_title(f"Predicted kappa_SU (lev={lev})")
        plt.colorbar(im1, ax=axes_su[1, i])

    fig_su.suptitle(f"kappa_SU: True vs Predicted (Time {time_idx})", fontsize=16)
    fig_su.tight_layout()

    # --- kappa_CA ---
    fig_ca, axes_ca = plt.subplots(2, num_levels, figsize=(5 * num_levels, 8))
    if num_levels == 1:
        axes_ca = np.expand_dims(axes_ca, axis=1)

    for i, lev in enumerate(lev_indices):
        im0 = axes_ca[0, i].imshow(y_true[1, lev], cmap='viridis')
        axes_ca[0, i].set_title(f"True kappa_CA (lev={lev})")
        plt.colorbar(im0, ax=axes_ca[0, i])

        im1 = axes_ca[1, i].imshow(y_pred[1, lev], cmap='viridis')
        axes_ca[1, i].set_title(f"Predicted kappa_CA (lev={lev})")
        plt.colorbar(im1, ax=axes_ca[1, i])

    fig_ca.suptitle(f"kappa_CA: True vs Predicted (Time {time_idx})", fontsize=16)
    fig_ca.tight_layout()

    return fig_su, fig_ca


def load_model_from_mlflow(run_id: str,
                         model_name: str = "model",
                         experiment_name: Optional[str] = None,
                         device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.nn.Module:
    """
    Load a PyTorch model from MLflow tracking server.
    """    
    
    # Set the tracking URI
    mlflow.set_tracking_uri("file:./mlruns")  # or your custom path
    
    # List all experiments for debugging
    client = MlflowClient()
    experiments = client.search_experiments()
    
    # Find the run
    try:
        run = client.get_run(run_id)

        # Load the model using the correct URI format
        model_uri = f"runs:/{run_id}/{model_name}"
        
        # Load the model
        model = mlflow.pytorch.load_model(model_uri, map_location=torch.device(device))
        model.eval()
        return model
        
    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        print("\nAvailable runs:")
        runs = client.search_runs(experiment_ids=[exp.experiment_id for exp in experiments])
        for r in runs:
            print(f"Run ID: {r.info.run_id}, Status: {r.info.status}, Start Time: {r.info.start_time}")
        raise

def visualize_mlflow_prediction(run_id: str,
                               nc_path: str,
                               time_idx: int = 0,
                               lev_indices: List[int] = [0],
                               model_name: str = "model",
                               experiment_name: Optional[str] = None,
                               device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                               save_path: Optional[str] = None) -> None:
    """
    Load a model from MLflow and visualize predictions.
    
    Args:
        run_id: MLflow run ID
        nc_path: Path to the NetCDF data file
        time_idx: Time index to visualize
        lev_indices: List of level indices to visualize
        model_name: Name used when saving the model in MLflow
        experiment_name: Optional name of the experiment
        device: Device to run the model on
        save_path: If provided, save the figure to this path
    """
    # Load the model
    model = load_model_from_mlflow(
        run_id=run_id,
        model_name=model_name,
        experiment_name=experiment_name,
        device=device
    )
    
    # Load dataset
    dataset = KappaDataset(nc_path)
    x, y_true = dataset[time_idx]
    x = x.unsqueeze(0).to(device)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        y_pred = model(x).squeeze(0).cpu().numpy()  # [n_classes, depth, height, width]
    
    y_true = y_true.numpy()
    
    # Set up subplots
    num_levels = len(lev_indices)
    fig, axes = plt.subplots(2, num_levels * 2, figsize=(5 * num_levels * 2, 10))
    
    if num_levels == 1:
        axes = np.expand_dims(axes, axis=1)
    
    # For each variable (kappa_SU and kappa_CA)
    for var_idx, var_name in enumerate(['kappa_SU', 'kappa_CA']):
        for i, lev in enumerate(lev_indices):
            # True values
            ax_true = axes[var_idx, i*2]
            im_true = ax_true.imshow(y_true[var_idx, lev], cmap='viridis')
            ax_true.set_title(f'True {var_name} - Level {lev}')
            plt.colorbar(im_true, ax=ax_true)
            
            # Predicted values
            ax_pred = axes[var_idx, i*2 + 1]
            im_pred = ax_pred.imshow(y_pred[var_idx, lev], cmap='viridis')
            ax_pred.set_title(f'Predicted {var_name} - Level {lev}')
            plt.colorbar(im_pred, ax=ax_pred)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    return fig
