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
        self.data = nc.Dataset(nc_path, 'r')
        total_timesteps = len(self.data.dimensions['time'])
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

        self.lev = self.data.dimensions['lev'].size
        self.lat = self.data.dimensions['lat'].size
        self.lon = self.data.dimensions['lon'].size

    def __len__(self):
        return len(self.indices)

    def _build_input_at_time(self, t: int) -> np.ndarray:
        inputs = []
        for var in self.inputs_3d:
            arr = np.nan_to_num(self.data.variables[var][t])  # [lat, lon]
            arr = np.tile(arr[None, :, :], (self.lev, 1, 1))   # [lev, lat, lon]
            inputs.append(arr)
        for var in self.inputs_4d:
            arr = np.nan_to_num(self.data.variables[var][t])  # [lev, lat, lon]
            inputs.append(arr)
        x = np.stack(inputs, axis=0)  # [C=12, lev, lat, lon]
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
