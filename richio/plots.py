
import importlib.resources as pkg_resources
import warnings
from typing import Any

import matplotlib.pyplot as plt
import typer
import unyt as u
import numpy as np

import richio.styles

# from richio.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

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
    with pkg_resources.path(richio.styles, "nice.mplstyle") as style_path:
        plt.style.use(style_path)


class SnapshotPlotter():

    def __init__(self, snap):

        self.data = snap
        self.peek = self.scatter        # alias for peek

    def scatter(self):
        pass

    def slice(self,
              data,
              ax=None,
              x='X',
              y='Y',
              res=512,
              method='scatter',
              **kwargs):
        pass
        

    def projection(self, 
                   data : str | u.unyt_array | np.ndarray, 
                   ax : Any | None = None, 
                   x : str | np.ndarray = 'X', 
                   y : str | np.ndarray = 'Y', 
                   res : Any | None = ...,
                   weights : str | u.unyt_array = 'volume',
                   method : str = 'hist',
                   **kwargs):
        """
        Make a projection plot.
        """
        if type(data) == str:
            key = data
            data = self.snap[key]
        elif type(data) == u.unyt_array:
            pass
        elif type(data) == np.ndarray:
            data = data * u.Dimensionless
            warnings.warn("No unit attached, assuming data is dimensionless.")
        else:
            raise TypeError(f"Data type {type(data)} unsupported."
                            "Use either str, unyt.unyt_array, or numpy.ndarray.")
        

        # Load coordinates
        if type(x) == str:
            x = self.snap[x]
        elif type(x) != u.unyt_array:
            raise TypeError(f"Data type for coordinates {type(data)} unsupported."
                            "Need unyt_array or str (column name).")


        # Volume or mass weighted or else
        if type(weights) == str:
            weights = self.snap[weights]
        elif type(x) != u.unyt_array:
            raise TypeError(f"Data type for weights {type(weights)} unsupported."
                            "Need unyt_array or str (column name).")


        # Calculate projection
        method_supported = ['hist']
        if method == method_supported[0]:
            H, xedges, yedges = np.histogram2d(x=x, y=y, bins=res, weights=data, density=False)
            X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
            ax.pcolormesh(X, Y, H, **kwargs)
        else:
            raise ValueError(f"method={method} is not supported."
                             "Currently supported: {method_supported}.")


        return ax        

    def slice():
        pass