import netCDF4 as nc
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Sequence


class SequenceKappaDataset(Dataset):
    """
    Time-window dataset returning sequences of inputs and the target at the last step.
    - Inputs per step are built identically to KappaDataset: 12 channels (4 tiled 3D + 8 4D).
    - Returns:
        x_seq: [T, C, lev, lat, lon]
        y_last: [2, lev, lat, lon] (targets at the last timestep of the window)
    """
    def __init__(self, nc_path: str, indices: Optional[Sequence[int]] = None, seq_len: int = 3):
        # Lazy open per worker
        self.nc_path = nc_path
        self.data = None
        with nc.Dataset(nc_path, 'r') as tmp:
            total_timesteps = len(tmp.dimensions['time'])
            self.lev = tmp.dimensions['lev'].size
            self.lat = tmp.dimensions['lat'].size
            self.lon = tmp.dimensions['lon'].size
        self.seq_len = int(seq_len)
        assert self.seq_len >= 1

        base_indices = np.arange(total_timesteps) if indices is None else np.asarray(indices)
        # Ensure indices are integers and sorted (TimeSeriesSplit provides sorted, contiguous indices)
        base_indices = np.sort(base_indices.astype(int))
        # Only keep indices that can form a full window ending at t
        self.indices = base_indices[base_indices >= (self.seq_len - 1)]

        # Variable lists consistent with KappaDataset
        self.inputs_3d = ['emi_SS', 'emi_OC', 'emi_BC', 'emi_DU']
        self.inputs_4d = ['DU_lite', 'SS_lite', 'SU_lite', 'CA_lite', 'apm1', 'svo', 'sd', 'st']
        self.target_vars = ['kappa_SU', 'kappa_CA']

        # Precompute per-channel normalization stats over a sample of timesteps
        self.input_mean, self.input_std = self._compute_input_stats(max_samples=256)

    def __len__(self):
        return len(self.indices)

    def _build_input_at_time(self, t: int) -> np.ndarray:
        # Ensure dataset is open in worker
        if self.data is None:
            self.data = nc.Dataset(self.nc_path, 'r')
            for v in list(self.inputs_3d) + list(self.inputs_4d) + list(self.target_vars):
                try:
                    self.data.variables[v].set_auto_maskandscale(False)
                except Exception:
                    pass
        inputs = []
        for var in self.inputs_3d:
            arr = np.nan_to_num(self.data.variables[var][t])  # [lat, lon]
            arr = np.tile(arr[None, :, :], (self.lev, 1, 1))   # [lev, lat, lon]
            inputs.append(arr)
        for var in self.inputs_4d:
            arr = np.nan_to_num(self.data.variables[var][t])  # [lev, lat, lon]
            inputs.append(arr)
        x = np.stack(inputs, axis=0)  # [C=12, lev, lat, lon]
        # Normalize per channel if stats available
        if self.input_mean is not None and self.input_std is not None:
            mean = self.input_mean.reshape(-1, 1, 1, 1)
            std = self.input_std.reshape(-1, 1, 1, 1)
            x = (x - mean) / std
        return x

    def __getitem__(self, idx):
        t = int(self.indices[idx])
        start_t = t - (self.seq_len - 1)
        # Build sequence [start_t, ..., t]
        x_steps = [self._build_input_at_time(tt) for tt in range(start_t, t + 1)]
        x_seq = np.stack(x_steps, axis=0)  # [T, C, lev, lat, lon]

        # Target at last timestep
        targets = [np.nan_to_num(self.data.variables[var][t]) for var in self.target_vars]
        y_last = np.stack(targets, axis=0)  # [2, lev, lat, lon]

        x_tensor = torch.tensor(x_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_last, dtype=torch.float32)
        return x_tensor, y_tensor

    def __del__(self):
        try:
            if self.data is not None:
                self.data.close()
        except Exception:
            pass

    def _compute_input_stats(self, max_samples: int = 256, log_normal: bool = True):
        """
        Compute per-channel statistics for the input channels using up to max_samples timesteps.
        
        Args:
            max_samples: Maximum number of timesteps to use for statistics computation
            log_normal: If True, compute statistics in log space (for log-normal distribution).
                       If False, compute normal distribution statistics.
                       Default is True (log-normal).
        """
        try:
            # Construct base indices able to form full windows
            # We compute stats on the inputs alone; windowing not needed.
            with nc.Dataset(self.nc_path, 'r') as ds:
                # disable mask/scale
                for v in list(self.inputs_3d) + list(self.inputs_4d):
                    try:
                        ds.variables[v].set_auto_maskandscale(False)
                    except Exception:
                        pass

                total_timesteps = len(ds.dimensions['time'])
                base_idxs = np.arange(total_timesteps)
                if len(base_idxs) == 0:
                    return None, None
                if len(base_idxs) > max_samples:
                    sel = np.linspace(0, len(base_idxs) - 1, num=max_samples, dtype=int)
                    base_idxs = base_idxs[sel]

                C = len(self.inputs_3d) + len(self.inputs_4d)
                sum_c = np.zeros(C, dtype=np.float64)
                sumsq_c = np.zeros(C, dtype=np.float64)
                count = 0

                for t in base_idxs:
                    inputs = []
                    for var in self.inputs_3d:
                        arr = np.nan_to_num(ds.variables[var][int(t)])  # [lat, lon]
                        arr = np.tile(arr[None, :, :], (self.lev, 1, 1))  # [lev, lat, lon]
                        inputs.append(arr)
                    for var in self.inputs_4d:
                        arr = np.nan_to_num(ds.variables[var][int(t)])  # [lev, lat, lon]
                        inputs.append(arr)
                    x = np.stack(inputs, axis=0)
                    x_reshaped = x.reshape(C, -1)
                    
                    if log_normal:
                        # For log-normal, we take log of data first (add small value to avoid log(0))
                        x_log = np.log(x_reshaped + 1e-12)
                        sum_c += x_log.sum(axis=1)
                        sumsq_c += (x_log ** 2).sum(axis=1)
                    else:
                        # Original normal distribution calculation
                        sum_c += x_reshaped.sum(axis=1)
                        sumsq_c += (x_reshaped ** 2).sum(axis=1)
                        
                    count += x_reshaped.shape[1]

                mean = sum_c / max(1, count)
                var = (sumsq_c / max(1, count)) - (mean ** 2)
                var = np.maximum(var, 1e-12)
                std = np.sqrt(var)
                
                if log_normal:
                    # Convert log-space mean and std back to original space
                    # For log-normal distribution:
                    # mu = exp(mean_log + std_log²/2)
                    # sigma² = (exp(std_log²) - 1) * exp(2*mean_log + std_log²)
                    mu = np.exp(mean + 0.5 * var)
                    sigma = np.sqrt((np.exp(var) - 1.0) * np.exp(2 * mean + var))
                    return mu.astype(np.float32), sigma.astype(np.float32)
                else:
                    return mean.astype(np.float32), std.astype(np.float32)
        except Exception:
            return None, None

