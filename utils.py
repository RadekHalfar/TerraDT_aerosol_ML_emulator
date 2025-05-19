import torch
import matplotlib.pyplot as plt
from model_CNN_test import KappaPredictorCNN
from KappaDataset import KappaDataset

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