import netCDF4 as nc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# --- Dataset ---
class KappaDataset(Dataset):
    def __init__(self, nc_path):
        self.data = nc.Dataset(nc_path, 'r')
        #self.timesteps = self.data.variables['time'].shape[0]
        self.timesteps = len(self.data.dimensions['time'])

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
        return self.timesteps

    def __getitem__(self, idx):
        inputs = []

        # 3D variables - broadcast over lev
        for var in self.inputs_3d:
            arr = self.data.variables[var][idx]  # (lat, lon)
            arr = np.nan_to_num(arr)
            arr = np.tile(arr[None, :, :], (self.lev, 1, 1))  # (lev, lat, lon)
            inputs.append(arr)

        # 4D variables
        for var in self.inputs_4d:
            arr = self.data.variables[var][idx]  # (lev, lat, lon)
            arr = np.nan_to_num(arr)
            inputs.append(arr)

        x = np.stack(inputs, axis=0)  # (channels, lev, lat, lon)

        # Targets
        targets = []
        for var in self.target_vars:
            y = self.data.variables[var][idx]  # (lev, lat, lon)
            y = np.nan_to_num(y)
            targets.append(y)

        y = np.stack(targets, axis=0)  # (2, lev, lat, lon)

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
def train_model(nc_file_path, epochs=2, batch_size=4, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    dataset = KappaDataset(nc_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #model = KappaPredictorCNN(in_channels=13).to(device)
    model = KappaPredictorCNN(in_channels=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")

    # Optional: Save model
    torch.save(model.state_dict(), "kappa_predictor_cnn.pth")
    print("✅ Training complete. Model saved to 'kappa_predictor_cnn.pth'.")

if __name__ == '__main__':
    train_model(r'C:\Users\radek\Documents\IT4I_projects\TerraDT\aerosol_ML_emulator\hamlite_sample_data_filtered.nc')
