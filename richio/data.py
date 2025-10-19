"""
Contains the main class Snapshot (SnapshotH5 and SnapshotNPY)

By: Yujie He
"""
import os
import warnings

import h5py
import numpy as np
import unyt as u

from richio.units import units
from richio.plots import SnapshotPlotter


def load(path):
    if os.path.isfile(path) and ((path[-2:] == 'h5') or (path[-4:] == 'hdf5')): # if hdf5 file
        with h5py.File(path) as f:
            pass
        return SnapshotH5(path)
    elif os.path.isdir(path): # if a numpy path
        return SnapshotNPY(path)
    else:
        raise FileNotFoundError(f"{path} is neither a directory nor a file.")


class Snapshot:

    def __init__(self, path):

        self.path = path
        self.plots = SnapshotPlotter(self)       # initialise a plotter object
        self._warnings_shown = set()            # Track which warnings we've shown

        # one-time warning of star consistency issue
        self._has_non_stars = self._should_filter_stars()
        if self._has_non_stars and ('star_filter_warning' not in self._warnings_shown):
            warnings.warn(
                "Non-star spurious gas detected. Consider using only star particles "
                "with: data = obj[obj.star_ratio_filter()]"
            )
            self._warnings_shown.add('star_filter_warning')

    


    


class SnapshotH5(Snapshot):   #TODO one class for direct output, one for merged, one for Paola/Konstantinos style npy dir
    
    def __init__(self, path):

        self.path = path
        self.rank = self.get_rank()             # after setting path
        
        super().__init__(path)                  # inherit all methods from parent class


    def get_rank(self) -> int:
        """
        Get the number of ranks.
        """
        with h5py.File(self.path, 'r') as f:
            maxrank = 0
            for key in f.keys():
                if key[:4] == 'rank':
                    rank = int(key[4:])
                    if maxrank < rank:
                        maxrank = rank

        maxrank += 1        # number of rank (starts from 1) is max rank (starts from 0) + 1
        return maxrank
        
    
    def __getitem__(self, key) -> u.unyt_array:     
        """
        Get numpy array of the desired quantity, combining different ranks.
        Supports slicing: obj['field'] or obj['field', 1:10] or obj['field', ::-1]
        """
        # Parse key and slice
        if isinstance(key, tuple):
            field, idx = key[0], key[1]
        else:
            field, idx = key, slice(None)
        
        with h5py.File(self.path, 'r') as f:
            try:
                arr = np.concatenate([f[f'rank{i}/{field}'] for i in range(self.rank)])
                arr *= units.get_unit(field)
            except KeyError:  # If field is not under rank, try on the root order
                arr = f[field][:] * units.get_unit(field)
        
        return arr[idx]


    def __len__(self) -> int:
        """
        Number of particles of the snapshot, combining different snapshots.
        """
        np = 0
        with h5py.File(self.path, 'r') as f:
            for i in range(self.rank):
                rankgroup = f[f'rank{i}']
                for key in rankgroup.keys():
                    np += len(rankgroup[key])
                    break
        return np
    

    def keys(self) -> list:

        with h5py.File(self.path, 'r') as f:
            keys = []
            for key in list(f.keys()):         # keys that are not in the ranks, usually 'Time', 'Box'
                if 'rank' in key:
                    continue
                else:
                    keys.append(key)
            
            keys += list(f['rank0'].keys())    # append keys under the ranks

        return keys
    

    def _should_filter_stars(self) -> bool:
        """
        Check if dataset has non-star particles.
        """
        with h5py.File(self.path, 'r') as f:
            for i in range(self.rank):
                if np.any(f[f'rank{i}/tracers/Star'][:] < 1 - 1e-3): 
                    return True
                else:
                    continue

        return False
    

    def star_ratio_filter(self) -> np.ndarray:
        """
        Get mask for star particles only (tracers/Star = 1).
        """
        return np.abs(self['tracers/Star'] - 1) < 1e-3
    
    

class SnapshotNPY(Snapshot):
    """
    Loading Paola's .npy directories.
    """

    def __init__(self, path):
        self.path = path
        self.snapnum = int(path[-2:])           # snapshot number
        super().__init__(path)                  # 
    

    def keys(self) -> list:

        files = os.listdir(self.path)           # files should look like Mass_30.npy, 30 being the snapshot number
        for i in range(len(files)):              
            _ = files[i].find('_')
            files[i] = files[i][:_]

        return files


    def __getitem__(self, key) -> u.unyt_array:     
        """
        Get numpy array of the desired quantity, combining different ranks.
        Supports slicing: obj['field'] or obj['field', 1:10] or obj['field', ::-1]
        """
        # Parse key and slice
        if isinstance(key, tuple):
            field, idx = key[0], key[1]
        else:
            field, idx = key, slice(None)
 
        filename = os.path.join(self.path, field+f'_{self.snapnum}.npy')  # format of extractor.py
        if os.path.isfile(filename):
            arr = np.load(filename, mmap_mode='r')      # try .npy
        else:                                           # if not .npy try .txt
            filename = os.path.join(self.path, field+f'_{self.snapnum}.txt')  # format of extractor.py
            if os.path.isfile(filename):
                arr = np.loadtxt(filename)
            else:
                raise FileNotFoundError(f"File {filename} is not found. Key '{key}' does not exist.")
        
        return arr[idx] * units.get_unit(field)


    def __len__(self) -> int:
        for key in self.keys():
            length = len(self[key])
            break
        return length


    def _should_filter_stars(self) -> bool:
        """
        Check if dataset has non-star particles.
        """
        star = self['Star']
        if np.any(star < 1 - 1e-3):
            return True
        else:
            return False
    

    def star_ratio_filter(self) -> np.ndarray:
        """
        Get mask for star particles only (tracers/Star = 1).
        """
        return np.abs(self['Star'] - 1) < 1e-3
        