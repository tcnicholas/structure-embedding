from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d

from structure_embedding.data import Dataset, get_indices_and_name, pad_features
from structure_embedding.utils import closest_key_error


class StructureEmbedding(ABC):
    """
    Base class to encapsulate the embedding of topology descriptors using
    a dimensionality reduction algorithm. Subclasses must implement the
    `fit_transform` and `transform` methods.
    """

    def __init__(self, **kwargs):
        self.settings = kwargs.copy()
        self.fitted = False
        self.global_padding = None
        self.training_data: list[Dataset] = []
        self.projection_data: list[Dataset] = []
        self._training_lookup: dict[str, int] = {}
        self._projection_lookup: dict[str, int] = {}

    @abstractmethod
    def fit_transform(
        self, 
        X: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """ Fit the embedding algorithm on X and return the embedded points. """
        pass

    @abstractmethod
    def transform(
        self, 
        X: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """ Transform new data X into the embedding space. """
        pass

    def add_training_data(
        self,
        data: npt.NDArray[np.floating],
        name: str | None = None,
        smearing: float | None = None,
        info: dict[str, Any] | None = None,
        properties: dict[str, Any] | None = None
    ) -> None:
        """
        Add training data to the model.
        """

        self._add_subset(
            data,
            self.training_data,
            self._training_lookup,
            name=name,
            smearing=smearing,
            info=info,
            properties=properties,
            reset_fitted=True  # Reset the fitted flag for training data.
        )

    def add_projection_data(
        self,
        data: npt.NDArray[np.floating],
        name: str | None = None,
        smearing: float | None = None,
        info: dict[str, Any] | None = None,
        properties: dict[str, Any] | None = None
    ) -> None:
        """
        Add projection data to the model.
        """

        self._add_subset(
            data,
            self.projection_data,
            self._projection_lookup,
            name=name,
            smearing=smearing,
            info=info,
            properties=properties
        )

    def pad_all(self, target_features: int) -> None:
        """
        Pad all data subsets to have the same number of features.
        """

        for subset in self.training_data + self.projection_data:
            pad_features(subset, target_features)

    @property
    def full_training_data(self) -> npt.NDArray[np.floating]:
        """
        Get the full training data array by concatenating all training subsets.
        """

        if not self.training_data:
            raise ValueError("No training data has been added.")

        return np.concatenate(
            [subset.raw for subset in self.training_data],
            axis=0
        )

    @property
    def full_projection_data(self) -> npt.NDArray[np.floating]:
        """
        Get the projection data array by concatenating all projection subsets.
        """
        if not self.projection_data:
            raise ValueError("No projection data has been added.")
        return np.concatenate(
            [subset.raw for subset in self.projection_data],
            axis=0
        )

    def get_embedding(
        self,
        padd_all_to: int | None = None,
        and_projections: bool = True
    ) -> npt.NDArray[np.floating]:
        """
        Fit the embedding algorithm on training data and return the embedding.

        Parameters
        ----------
        padd_all_to
            Number of features to pad all subsets to.
        and_projections
            If True, also transform the projection data.

        Returns
        -------
        The embedding of the training data.
        """
        padd_all_to = 0 if padd_all_to is None else padd_all_to
        if (
            self.fitted and 
            self.global_padding is not None and 
            self.global_padding >= padd_all_to
        ):
            return np.vstack([
                subset.dim_red_data for subset in self.training_data
            ])

        if padd_all_to:
            self.pad_all(padd_all_to)

        self._sanity_check_data()
        X = self.full_training_data

        embedding = self.fit_transform(X)
        self.fitted = True

        for subset in self.training_data:
            subset.dim_red_data = embedding[
                subset.starting_ix:subset.ending_ix + 1
            ]

        if and_projections:
            self.project()

        return embedding

    def project(self) -> npt.NDArray[np.floating]:
        """
        Transform projection data using the fitted embedding algorithm.

        Returns
        -------
        The transformed projection data.
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted.")
        X = self.full_projection_data
        projections = self.transform(X)

        for subset in self.projection_data:
            subset.dim_red_data = projections[
                subset.starting_ix:subset.ending_ix + 1
            ]

        return projections

    def get_training_subset(self, name: str) -> Dataset:
        """
        Get a training subset by name.
        """
        if name not in self._training_lookup:
            closest_key_error(name, self._training_lookup.keys())
        return self.training_data[self._training_lookup[name]]

    def get_projection_subset(self, name: str) -> Dataset:
        """
        Get a projection subset by name.
        """
        if name not in self._projection_lookup:
            closest_key_error(name, self._projection_lookup.keys())
        return self.projection_data[self._projection_lookup[name]]

    def filter_by(
        self, 
        predicate: callable,
        dataset_type: str = "training"
    ):
        """
        Filter datasets based on a predicate function.

        Parameters
        ----------
        predicate
            A function that takes a Dataset instance and returns a boolean.
        dataset_type
            Specifies which dataset to search: "training" or "projected".

        Returns
        -------
        A generator yielding all subsets for which predicate(subset) is True.
        """
        if dataset_type == "training":
            data = self.training_data
        elif dataset_type == "projected":
            data = self.projection_data
        else:
            raise ValueError(
                "dataset_type must be either 'training' or 'projected'"
            )
        return (subset for subset in data if predicate(subset))

    def _add_subset(
        self,
        data: npt.NDArray[np.floating],
        container: list[Dataset],
        lookup: dict[str, int],
        name: str | None = None,
        smearing: float | None = None,
        info: dict[str, Any] | None = None,
        properties: dict[str, Any] | None = None,
        reset_fitted: bool = False,
    ) -> None:
        """
        Add a new data subset to the given container and update the 
        corresponding lookup.
        """
        if reset_fitted:
            self.fitted = False

        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            raise ValueError("Data must be 1D or 2D.")

        info = {} if info is None else info
        properties = {} if properties is None else properties
        starting_ix, ending_ix, candidate_name = get_indices_and_name(
            data, container, name
        )

        data = data.astype(np.float64)

        new_subset = Dataset(
            name=candidate_name,
            raw=data,
            starting_ix=starting_ix,
            ending_ix=ending_ix,
            dim_red_data=None,
            smearing=smearing,
            info=info,
            properties=properties
        )

        container.append(new_subset)
        lookup[candidate_name] = len(container) - 1

    def _sanity_check_data(self) -> None:
        """
        Check that training data has been added and that all subsets have the 
        same number of features. Apply smearing if specified.
        """

        if not self.training_data:
            raise ValueError("No training data has been added.")
        
        if len({subset.feature_size for subset in self.training_data}) != 1:

            max_features = max(
                subset.feature_size 
                for subset in self.training_data + self.projection_data
            )

            for subset in self.training_data:
                pad_features(subset, max_features)

            for subset in self.projection_data:
                pad_features(subset, max_features)

            self.global_padding = max_features

        for subset in self.training_data + self.projection_data:
            if subset.smearing is not None:
                subset.raw = gaussian_filter1d(
                    subset.raw.astype(np.float64), 
                    sigma=subset.smearing, 
                    axis=1
                )