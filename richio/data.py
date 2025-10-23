"""
Contains the main class Snapshot (SnapshotH5 and SnapshotNPY)

By: Yujie He
"""

import os
import re

import h5py
import numpy as np
import unyt as u

from richio.plots import SnapshotPlotter
from richio.units import units


def load(path):
    if os.path.isfile(path) and (path.endswith("h5") or path.endswith("hdf5")):  # if hdf5 file
        with h5py.File(path) as f:
            pass
        return SnapshotH5(path)
    elif os.path.isdir(path):  # if a directory
        return SnapshotNPY(path)
    else:
        raise FileNotFoundError(f"{path} is neither a directory nor a file.")


class Snapshot:
    def __init__(self, path: str):
        self.path = path
        self.plots = SnapshotPlotter(self)  # initialise a plotter object
        self.snapnum = self._get_snapnum()  # snapshot number

        # Build class mapping once (idempotent)
        if not self.__class__._alias_to_canonical:
            self._build_alias_mapping()

    def _get_snapnum(self):
        """
        Try to read the snapshot number from the path. Matches `snap_<num>`
        pattern across the path and returns the first math. Set to -1 if no
        match is found.
        """

        # pattern to match 'snap_' followed by digits
        pattern = r"snap_(\d+)"  # (...) matches whatever regular expressions inside the parenthesis, \d matches 0-9, and + matches many digits
        match = re.search(pattern, self.path)  # search for patterns all inside the path

        if match:
            return int(match.group(1))
        else:
            raise UserWarning(f"No snapshot number found in path: {self.path}")
            return -1

    @classmethod
    def _build_alias_mapping(cls):
        """Build once per class, not per instance."""
        for canonical, aliases in cls._field_aliases.items():
            cls._alias_to_canonical[canonical] = canonical
            for alias in aliases:
                cls._alias_to_canonical[alias] = canonical

    def __getattr__(self, name):
        """Allow attribute-style access: snap.density"""

        try:
            return self[name]
        except FileNotFoundError:
            raise AttributeError(f"Field '{name}' not found")

    def _resolve_field_name(self, key: str) -> str:
        """Read from shared class dict."""
        return self.__class__._alias_to_canonical.get(key, key)

    def mask_star_ratio(self) -> np.ndarray:
        """
        Get mask for star particles only (tracers/Star = 1).
        """
        return np.abs(self.star - 1) < 1e-3

    def mask_density(self) -> np.ndarray:
        """
        Get mask for floor density gas.
        """
        return self.density > 1e-19 * units.get_unit("Density")

    @property
    def _field_info(self):
        """
        Dictionary mapping field names to metadata.

        Returns
        -------
        dict
            Dictionary with field names as keys and metadata as values

        Examples
        --------
        >>> info = snap._field_info
        >>> print(info['Density'])
        {'unit': g/cm³, 'aliases': ['Den', 'density', 'rho']}
        """
        info = {}

        for field in self.keys():
            info[field] = {
                "unit": units.get_unit(field),
                "aliases": self._field_aliases[field],
            }

        return info

    def info(self, unit_system="rich", show_aliases=True) -> None:
        """
        Display snapshot information and available fields.

        Parameters
        ----------
        unit_system : str, optional
            Unit system: 'rich' (default), 'cgs', or 'mks'
        show_aliases : bool, optional
            Show field aliases (default: True)

        Examples
        --------
        >>> snap.info()
        >>> snap.info(unit_system='cgs')
        >>> snap.info(show_aliases=False)
        """
        # Header
        print("=" * 100)
        print(f"RICH SNAPSHOT INFORMATION".center(100))
        print("=" * 100)

        # Metadata
        print(f"\n{'Snapshot Details':<40}")
        print("-" * 100)

        meta_info = [
            ("Path", self.path),
            ("Snapshot Number", self.snapnum),
        ]

        # Add optional metadata
        for attr, label in [("time", "Time"), ("box", "Box size"), ("cycle", "Cycle")]:
            try:
                val = getattr(self, attr)
                meta_info.append((label, val.in_base(unit_system)))
            except:
                pass

        try:
            meta_info.append(("Number of Cells", f"{len(self):,}"))
        except:
            pass

        if hasattr(self, "rank"):
            meta_info.append(("Number of Ranks", self.rank))

        for label, value in meta_info:
            print(f"  {label:<25} : {value}")

        # Fields
        print(f"\n{'Available Fields':<40} [Unit System: {unit_system.upper()}]")
        print("-" * 100)

        # Header row
        if show_aliases:
            print(f"{'Field':<15} {'Unit':<40} {'Aliases'}")
        else:
            print(f"{'Field':<15} {'Unit'}")
        print("-" * 100)

        # Field information
        field_info_dict = self._field_info

        for field in self.keys():
            if field not in field_info_dict:
                continue

            info = field_info_dict[field]

            # Format unit
            if unit_system == "rich":
                unit_str = str(info["unit"])
            else:
                unit_str = str((1 * info["unit"]).in_base(unit_system))

            if show_aliases:
                aliases = info.get("aliases", [])
                alias_str = ", ".join(aliases) if aliases else "-"
                print(f"{field:<15} {unit_str:<40} {alias_str}")
            else:
                print(f"{field:<15} {unit_str}")

        # Footer
        print("-" * 100)
        print(f"Total: {len(self.keys())} fields")

        print("=" * 100)


class SnapshotH5(
    Snapshot
):  # TODO one class for direct output, one for merged, one for Paola/Konstantinos style npy dir
    _field_aliases = {
        # Positions
        "X": ["position_x", "pos_x", "x", "particle_position_x"],
        "Y": ["position_y", "pos_y", "y", "particle_position_y"],
        "Z": ["position_z", "pos_z", "z", "particle_position_z"],
        # Center of mass
        "CMx": ["cm_x", "center_of_mass_x"],
        "CMy": ["cm_y", "center_of_mass_y"],
        "CMz": ["cm_z", "center_of_mass_z"],
        # Velocities
        "Vx": ["velocity_x", "vx", "vel_x", "particle_velocity_x"],
        "Vy": ["velocity_y", "vy", "vel_y", "particle_velocity_y"],
        "Vz": ["velocity_z", "vz", "vel_z", "particle_velocity_z"],
        "divV": ["velocity_divergence", "DivV", "div_v", "divergence"],
        # Thermodynamic
        "Density": [
            "density",
            "densitites",
            "Den",
            "rho",
        ],  # swiftsimio: 'densities', yt: 'density'
        "Pressure": ["pressure", "P"],
        "Temperature": ["temperature", "T", "temp"],
        "InternalEnergy": ["internal_energy", "IE", "specific_internal_energy", "sie"],
        "tracers/Entropy": ["entropy", "Entropy", "S"],
        "Dissipation": ["dissipation", "Diss"],
        # Volume
        "Volume": ["volume", "Vol", "volumes"],
        # Radiation
        "Erad": ["radiation_energy", "Rad", "E_rad"],
        # Gradients - Pressure
        "DpDx": ["pressure_gradient_x", "grad_p_x", "dp_dx"],
        "DpDy": ["pressure_gradient_y", "grad_p_y", "dp_dy"],
        "DpDz": ["pressure_gradient_z", "grad_p_z", "dp_dz"],
        # Gradients - Density
        "DrhoDx": ["density_gradient_x", "grad_rho_x", "drho_dx"],
        "DrhoDy": ["density_gradient_y", "grad_rho_y", "drho_dy"],
        "DrhoDz": ["density_gradient_z", "grad_rho_z", "drho_dz"],
        # Gradients - Internal Energy
        "DsieDx": ["sie_gradient_x", "grad_sie_x", "dsie_dx"],
        "DsieDy": ["sie_gradient_y", "grad_sie_y", "dsie_dy"],
        "DsieDz": ["sie_gradient_z", "grad_sie_z", "dsie_dz"],
        # The percentage of material that comes from star
        "tracers/Star": ["star_fraction", "Star", "star", "star_ratio", "stellar_fraction"],
        # Metadata
        "Box": ["box_size", "box", "boxsize"],
        "Time": ["time", "t", "tfb", "simulation_time", "current_time"],
        "Cycle": ["cycle", "step", "iteration"],
        "ID": ["particle_id", "id", "ids", "particle_ids"],
        # Unused
        "Eg_0": ["eg_0"],
        "stickers": ["stickers"],
        "tracers/WasRemoved": ["was_removed", "WasRemoved", "removed"],
    }

    # Reverse mapping for quick lookup, build only once
    _alias_to_canonical = {}

    def __init__(self, path):
        self.path = path
        self.rank = self._get_rank()  # after setting path

        super().__init__(path)  # inherit all methods from parent class

    def _get_rank(self) -> int:
        """
        Get the number of ranks.
        """
        with h5py.File(self.path, "r") as f:
            maxrank = 0
            for key in f.keys():
                if key[:4] == "rank":
                    rank = int(key[4:])
                    if maxrank < rank:
                        maxrank = rank

        maxrank += 1  # number of rank (starts from 1) is max rank (starts from 0) + 1
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

        # Resolve alias to canonical name (class method, no instance data)
        field = self._resolve_field_name(field)

        with h5py.File(self.path, "r") as f:
            try:
                arr = np.concatenate([f[f"rank{i}/{field}"] for i in range(self.rank)])
                arr *= units.get_unit(field)
            except KeyError:  # If field is not under rank, try on the root order
                arr = f[field][:] * units.get_unit(field)

        return arr[idx]

    def __len__(self) -> int:
        """
        Number of particles of the snapshot, combining different snapshots.
        """
        np = 0
        with h5py.File(self.path, "r") as f:
            for i in range(self.rank):
                rankgroup = f[f"rank{i}"]
                for key in rankgroup.keys():
                    np += len(rankgroup[key])
                    break
        return np

    def keys(self) -> list:  # Update this to recursively list all datasets
        with h5py.File(self.path, "r") as f:
            keys = []
            for key in list(f.keys()):  # keys that are not in the ranks, usually 'Time', 'Box'
                if key.startswith("rank"):
                    continue
                else:
                    keys.append(key)

            keys += list(f["rank0"].keys())  # append keys under the ranks

        keys.sort()

        return keys


class SnapshotNPY(Snapshot):
    """
    Loading Paola's .npy directories.
    """

    _field_aliases = {
        # Positions
        "CMx": SnapshotH5._field_aliases["CMx"],
        "CMy": SnapshotH5._field_aliases["CMy"],
        "CMz": SnapshotH5._field_aliases["CMz"],
        # Velocities
        "Vx": SnapshotH5._field_aliases["Vx"],
        "Vy": SnapshotH5._field_aliases["Vy"],
        "Vz": SnapshotH5._field_aliases["Vz"],
        "DivV": SnapshotH5._field_aliases["divV"],
        # Mass and volume
        "Mass": [
            "mass",
            "masses",
            "particle_mass",
            "m",
        ],  # swiftsimio uses 'masses' # calculated by den*vol, not directly from output .h5
        "Vol": SnapshotH5._field_aliases["Volume"],
        # Thermodynamic
        "Den": SnapshotH5._field_aliases["Density"],
        "P": SnapshotH5._field_aliases["Pressure"],
        "T": SnapshotH5._field_aliases["Temperature"],
        "IE": SnapshotH5._field_aliases["InternalEnergy"],
        "Diss": SnapshotH5._field_aliases["Dissipation"],
        "Entropy": SnapshotH5._field_aliases["tracers/Entropy"],
        # Radiation
        "Rad": SnapshotH5._field_aliases["Erad"],
        # Gradients - Pressure
        "DpDx": SnapshotH5._field_aliases["DpDx"],
        "DpDy": SnapshotH5._field_aliases["DpDy"],
        "DpDz": SnapshotH5._field_aliases["DpDz"],
        # The star fraction
        "Star": SnapshotH5._field_aliases["tracers/Star"],
        # Metadata
        "box": SnapshotH5._field_aliases["Box"],
        "tfb": SnapshotH5._field_aliases["Time"],
    }

    # Reverse mapping for quick lookup, build only once
    _alias_to_canonical = {}

    def __init__(self, path):
        self.path = path  # TODO: rewrite with Path

        super().__init__(path)

    def keys(self) -> list:  # TODO: rewrite this with re
        keys = []
        files = os.listdir(
            self.path
        )  # files should look like Mass_30.npy, 30 being the snapshot number
        for i in range(len(files)):
            if files[i].endswith("txt") or files[i].endswith("npy"):
                _ = files[i].find("_")
                keys.append(files[i][:_])
            else:
                continue

        keys.sort()

        return keys

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

        # Resolve alias to canonical name (class method, no instance data)
        field = self._resolve_field_name(field)

        filename = os.path.join(
            self.path, field + f"_{self.snapnum}.npy"
        )  # format of extractor.py
        if os.path.isfile(filename):
            arr = np.load(filename, mmap_mode="r")  # try .npy
        else:  # if not .npy try .txt
            filename = os.path.join(
                self.path, field + f"_{self.snapnum}.txt"
            )  # format of extractor.py
            if os.path.isfile(filename):
                arr = np.loadtxt(filename)
            else:
                raise FileNotFoundError(
                    f"File {filename} is not found. Key '{key}' does not exist."
                )

        if idx == slice(None):  # do not slice if no index indicated
            return arr * units.get_unit(
                field
            )  # slice[idx] works generally so long as arr is not 0-dimensional
        else:  # which is the case for snap.time
            return arr[idx] * units.get_unit(field)

    def __len__(self) -> int:
        for key in self.keys():
            length = len(self[key])
            break
        return length
