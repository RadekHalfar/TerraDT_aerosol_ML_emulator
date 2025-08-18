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

        # Precompute per-channel normalization stats (inputs only) over a sample of timesteps
        self.input_mean, self.input_std = self._compute_input_stats(max_samples=256)

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
        # Normalize per channel: (x - mean) / std
        if self.input_mean is not None and self.input_std is not None:
            # reshape for broadcasting: [C, 1, 1, 1]
            mean = self.input_mean.reshape(-1, 1, 1, 1)
            std = self.input_std.reshape(-1, 1, 1, 1)
            x = (x - mean) / std
        targets = [np.nan_to_num(self.data.variables[var][t]) for var in self.target_vars]
        y = np.stack(targets, axis=0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __del__(self):
        try:
            if self.data is not None:
                self.data.close()
        except Exception:
            pass

    def _compute_input_stats(self, max_samples: int = 256):
        """
        Compute per-channel mean and std for the 12 input channels using up to max_samples timesteps.
        Stats are computed over all spatial positions and sampled times.
        """
        try:
            # Choose sample timesteps uniformly across available indices
            idxs = np.asarray(self.indices)
            if len(idxs) == 0:
                return None, None
            if len(idxs) > max_samples:
                # uniform subsample
                sel = np.linspace(0, len(idxs) - 1, num=max_samples, dtype=int)
                idxs = idxs[sel]

            # Running sums for mean and variance per channel
            C = len(self.inputs_3d) + len(self.inputs_4d)
            sum_c = np.zeros(C, dtype=np.float64)
            sumsq_c = np.zeros(C, dtype=np.float64)
            count = 0

            with nc.Dataset(self.nc_path, 'r') as ds:
                # disable mask/scale
                for v in list(self.inputs_3d) + list(self.inputs_4d):
                    try:
                        ds.variables[v].set_auto_maskandscale(False)
                    except Exception:
                        pass
                for t in idxs:
                    inputs = []
                    for var in self.inputs_3d:
                        arr = np.nan_to_num(ds.variables[var][int(t)])  # [lat, lon]
                        arr = np.tile(arr[None, :, :], (self.lev, 1, 1))  # [lev, lat, lon]
                        inputs.append(arr)
                    for var in self.inputs_4d:
                        arr = np.nan_to_num(ds.variables[var][int(t)])  # [lev, lat, lon]
                        inputs.append(arr)
                    x = np.stack(inputs, axis=0)  # [C, lev, lat, lon]
                    # accumulate per channel
                    x_reshaped = x.reshape(C, -1)
                    sum_c += x_reshaped.sum(axis=1)
                    sumsq_c += (x_reshaped ** 2).sum(axis=1)
                    count += x_reshaped.shape[1]

            mean = sum_c / max(1, count)
            var = (sumsq_c / max(1, count)) - (mean ** 2)
            var = np.maximum(var, 1e-12)
            std = np.sqrt(var)
            return mean.astype(np.float32), std.astype(np.float32)
        except Exception:
            # On any failure, skip normalization rather than breaking training
            return None, None