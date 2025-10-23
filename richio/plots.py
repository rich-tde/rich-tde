from importlib.resources import files
from typing import Any
import warnings

from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import unyt as u

# from richio.config import FIGURES_DIR, PROCESSED_DATA_DIR

# import typer
# app = typer.Typer()

# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     output_path: Path = FIGURES_DIR / "plot.png",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Generating plot from data...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Plot generation complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()


def use_nice_style():
    style_path = files("richio.styles").joinpath("nice.mplstyle")
    plt.style.use(style_path)


class SnapshotPlotter:
    def __init__(self, snap):
        self.snap = snap
        self.peek = self.scatter  # alias for peek

    def scatter(self):
        pass

    def slice(self, data, ax=None, x="X", y="Y", res=512, method="scatter", **kwargs):
        pass

    # TODO extract to a new function
    def projection(
        self,
        data: str | ArrayLike,
        res: int | ArrayLike,
        x: str | ArrayLike = "X",
        y: str | ArrayLike = "Y",
        z: str | ArrayLike = "Z",
        ax: Any | None = None,
        unit_system: str = "cgs",
        box_size: ArrayLike | None = None,
        cmap: str | Colormap = "twilight",  # default to twilight, because it looks nice
        star_mask: bool = True,
        **kwargs,
    ):
        """
        Make a projection plot. To make use of the unit system, use either str
        keys or unyt_array data for `data`, `x`, `y`, `z`, `box_size`.
        """
        from scipy.spatial import KDTree

        # Fetch data
        data = self._get_data(data)
        x = self._get_data(x)
        y = self._get_data(y)
        z = self._get_data(z)

        # Set boxsize
        if box_size is None:
            x0, y0, z0, x1, y1, z1 = self.snap.box  # Load the box size
        else:
            x0, y0, z0, x1, y1, z1 = box_size

        # Set resolution
        try:
            nx, ny, nz = res[0], res[1], res[2]
        except TypeError:
            nx = ny = nz = res

        # Make Euclidean grid
        xspace = np.linspace(
            x0, x1, nx, endpoint=False
        )  # disable endpoints such that dz = (z1-z0)/res instead of (z1-z0)/(res-1)
        yspace = np.linspace(y0, y1, ny, endpoint=False)
        zspace = np.linspace(z0, z1, nz, endpoint=False)  # TODO: add an option to use np.geomspace

        X, Y, Z = np.meshgrid(
            xspace, yspace, zspace, indexing="ij"
        )  # make the grid for interpolation

        coords = np.stack([x, y, z], axis=-1)  # coordinates of the particles
        grid_coords = np.stack([X, Y, Z], axis=-1)  # coordinates of the grid (query points)

        tree = KDTree(coords)  # build tree
        d, i = tree.query(
            grid_coords, k=1, eps=0.0, p=2, workers=1
        )  # the most time-consuming step        # TODO: allow increased workers for parallel processing

        grid_data = data[i]

        dz = (z1 - z0) / nz
        projected_data = np.sum(grid_data * dz, axis=-1).in_base(unit_system)

        if ax is None:
            fig, ax = plt.subplots()

        # Plot
        xx, yy = np.meshgrid(xspace, yspace, indexing="ij")
        im = ax.pcolormesh(xx, yy, np.log10(projected_data), cmap=cmap, **kwargs)
        unit_latex = projected_data.units.latex_repr
        plt.colorbar(im, ax=ax, label=f"$\\log(\\Sigma/{unit_latex})$")

        return ax, projected_data

    def _get_data(
        self, data: str | ArrayLike
    ) -> u.unyt_array:  # TODO: add masking option that automatically mask floor gas
        if isinstance(data, str):
            key = data
            data = self.snap[key]
        elif isinstance(data, u.unyt_array):
            pass
        elif isinstance(data, np.ndarray):
            data = data * u.Dimensionless
            warnings.warn("No unit attached, assuming data is dimensionless.")
        else:
            raise TypeError(
                f"Data type {type(data)} unsupported."
                "Use either str, unyt.unyt_array, or numpy.ndarray."
            )

        return data
