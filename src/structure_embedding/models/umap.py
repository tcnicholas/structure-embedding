from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import umap

from structure_embedding.model import StructureEmbedding

_UMAP_DEFAULTS = {
    "n_neighbors": 50,
    "min_dist": 0.25,
    "metric": "euclidean",
    "n_jobs": 1,
    "random_state": 42,
}

class UMAPStructureEmbedding(StructureEmbedding):
    """
    A implementation of StructureEmbedding using UMAP.
    """

    def __init__(self, **kwargs):
        umap_kwargs = _UMAP_DEFAULTS.copy() | kwargs
        self._embedder = umap.UMAP(**umap_kwargs)
        super().__init__(**umap_kwargs)

    def fit_transform(
        self, 
        X: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            embedding = self._embedder.fit_transform(X)

        return embedding

    def transform(
        self, 
        X: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            projections = self._embedder.transform(X)
        return projections