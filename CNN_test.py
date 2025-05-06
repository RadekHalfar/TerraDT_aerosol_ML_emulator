import netCDF4 as nc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Dataset ---
class KappaDataset(Dataset):
    def __init__(self, nc_path, indices=None):
        self.data = nc.Dataset(nc_path, 'r')
        total_timesteps = len(self.data.dimensions['time'])

        self.indices = indices if indices is not None else list(range(total_timesteps))

        self.inputs_3d = ['emi_SS', 'emi_OC', 'emi_BC', 'emi_DU']
        self.inputs_4d = [
            'DU_lite', 'SS_lite', 'SU_lite', 'CA_lite',
            'apm1', 'svo', 'sd', 'st'
        ]
        self.target_vars = ['kappa_SU', 'kappa_CA']

        self.lev = self.data.dimensions['lev'].size
        self.lat = self.data.dimensions['lat'].size
        self.lon = self.data.dimensions['lon'].size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        inputs = []

        for var in self.inputs_3d:
            arr = np.nan_to_num(self.data.variables[var][t])
            arr = np.tile(arr[None, :, :], (self.lev, 1, 1))
            inputs.append(arr)

        for var in self.inputs_4d:
            arr = np.nan_to_num(self.data.variables[var][t])
            inputs.append(arr)

        x = np.stack(inputs, axis=0)

        targets = [
            np.nan_to_num(self.data.variables[var][t])
            for var in self.target_vars
        ]
        y = np.stack(targets, axis=0)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)



# --- Model ---
class KappaPredictorCNN(nn.Module):
    def __init__(self, in_channels):
        super(KappaPredictorCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 2, kernel_size=1)  # Output: kappa_SU, kappa_CA
        )

    def forward(self, x):  # x: [B, C, L, H, W]
        return self.decoder(self.encoder(x))


# --- Training ---
def train_model(nc_file_path, epochs=3, batch_size=4, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
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

    model = KappaPredictorCNN(in_channels=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
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


    # Plot learning curves
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot train loss on left y-axis
    ax1.plot(train_losses, color='blue', label='Train Loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss (MSE)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(val_losses, color='red', label='Validation Loss')
    ax2.set_ylabel("Validation Loss (MSE)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Title and legend
    plt.title("Training Curve")
    fig.tight_layout()
    plt.show()

    # Save model
    torch.save(model.state_dict(), "kappa_predictor_cnn.pth")
    print("✅ Training complete. Model saved.")


if __name__ == '__main__':
    train_model(r'C:\Users\radek\Documents\IT4I_projects\TerraDT\aerosol_ML_emulator\hamlite_sample_data_filtered.nc')
