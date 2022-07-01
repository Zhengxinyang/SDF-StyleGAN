import torch
from torch.utils import data
from pathlib import Path
import numpy as np

sdf_EXTS = ['npy']

class sdfDataset(data.Dataset):
  def __init__(self, folder, truncated:bool=False):
    super().__init__()
    self.folder = folder
    self.truncated = truncated
    self.paths = [p for ext in sdf_EXTS for p in Path(
        f'{folder}').glob(f'**/*.{ext}')]
    assert len(self.paths) > 0, f'No data were found in {folder} for training'

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, index):
    _path = self.paths[index]
    _data = torch.from_numpy(np.load(_path))

    if self.truncated:
      _data = torch.clamp(_data, min=-0.1, max=0.1)

    return _data[None, :, :, :]
