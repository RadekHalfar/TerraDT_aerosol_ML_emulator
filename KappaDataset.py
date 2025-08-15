import netCDF4 as nc
import numpy as np
import torch
from torch.utils.data import Dataset

class KappaDataset(Dataset):
    def __init__(self, nc_path, indices=None):
        # Store path and lazily open inside workers to enable num_workers>0
        self.nc_path = nc_path
        self.data = None  # opened on first __getitem__ per worker
        # We need basic dims to compute indices; open temp handle
        with nc.Dataset(nc_path, 'r') as tmp:
            total_timesteps = len(tmp.dimensions['time'])
            self.lev = tmp.dimensions['lev'].size
            self.lat = tmp.dimensions['lat'].size
            self.lon = tmp.dimensions['lon'].size
        self.indices = indices if indices is not None else list(range(total_timesteps))
        self.inputs_3d = ['emi_SS', 'emi_OC', 'emi_BC', 'emi_DU']
        self.inputs_4d = ['DU_lite', 'SS_lite', 'SU_lite', 'CA_lite', 'apm1', 'svo', 'sd', 'st']
        self.target_vars = ['kappa_SU', 'kappa_CA']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Lazy open per worker
        if self.data is None:
            self.data = nc.Dataset(self.nc_path, 'r')
            # Disable auto mask/scale for speed
            for v in list(self.inputs_3d) + list(self.inputs_4d) + list(self.target_vars):
                try:
                    self.data.variables[v].set_auto_maskandscale(False)
                except Exception:
                    pass
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
        targets = [np.nan_to_num(self.data.variables[var][t]) for var in self.target_vars]
        y = np.stack(targets, axis=0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __del__(self):
        try:
            if self.data is not None:
                self.data.close()
        except Exception:
            pass