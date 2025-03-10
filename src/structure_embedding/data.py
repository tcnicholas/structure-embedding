from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from structure_embedding.utils import uniform_repr


@dataclass
class Dataset:
    """
    Store (a subset of) training data that will be used to fit the UMAP model.
    """
    name: str
    """ The name of the data subset. """

    raw: npt.NDArray[np.floating]
    """ The raw data used to fit the UMAP model. """

    starting_ix: int
    """ The index of the first data point within the full training data. """

    ending_ix: int
    """ The index of the last data point within the full training data. """

    dim_red_data: npt.NDArray[np.floating] | None
    """ The dimensionally reduced data (initially not computed). """

    smearing: float | None = None
    """ The standard deviation of the Gaussian smearing applied to the data. """

    info: dict[str, Any] = field(default_factory=dict)
    """ Additional information to store with the dataset. """

    properties: dict[str, Any] = field(default_factory=dict)
    """ Additional properties to store with the dataset. """

    @property
    def n_points(self) -> int:
        """ The number of data points in the subset. """

        return self.raw.shape[0]
    
    @property
    def feature_size(self) -> int:
        """ The number of features in the data. """

        return self.raw.shape[1]

    def __repr__(self) -> str:
        """ Generate a string representation of the dataset. """

        info = {
            "name": f'"{self.name}"',
            "n_points": self.n_points,
            "fitted": self.dim_red_data is not None,
            "info": uniform_repr(
                "", 
                **self.info, 
                indent_size=4, 
                stringify=True
            ),
            "properties": uniform_repr(
                "", 
                **self.properties, 
                indent_size=4, 
                stringify=True
            )
        }

        if self.smearing is not None:
            info["smearing"] = np.round(self.smearing, 2)

        return uniform_repr("Dataset", **info, indent_size=4, stringify=False)
    

########## HELPER FUNCTIONS ##########
    

def pad_features(subset: Dataset, target_features: int) -> None:
    """
    Pad the given subset's raw data with zeros so that it has the desired number
    of features.
    """
    current_features = subset.feature_size
    if current_features < target_features:
        padding = np.zeros((subset.n_points, target_features-current_features))
        subset.raw = np.concatenate([subset.raw, padding], axis=1)


def get_indices_and_name(
    data: npt.NDArray[np.floating],
    collection: list[Dataset],
    name: str | None = None,
    default_prefix: str = "subset"
) -> tuple[int, int, str]:
    """
    Compute the starting and ending indices for the new data subset, and 
    determine a unique name for it.
    """

    current_length = (collection[-1].ending_ix + 1) if collection else 0
    new_length = data.shape[0]
    starting_ix = current_length
    ending_ix = current_length + new_length - 1

    candidate_name = (
        name if name is not None else f"{default_prefix}_{len(collection)}"
    )
    existing_names = {item.name for item in collection}
    if candidate_name in existing_names:
        counter = 1
        new_candidate_name = f"{candidate_name}-{counter}"
        while new_candidate_name in existing_names:
            counter += 1
            new_candidate_name = f"{candidate_name}-{counter}"

        warnings.warn(
            f"Name '{candidate_name}' already exists in the dataset. Using "
            f"'{new_candidate_name}' instead.",
            stacklevel=2
        )
        candidate_name = new_candidate_name

    return starting_ix, ending_ix, candidate_name