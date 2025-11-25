from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import datetime
import numpy as np
import pandas as pd
# import lightning as pl
import tsl
from typing import Optional

from tsl.datasets.prototypes import DatetimeDataset, casting
from tsl.ops import similarities as sims


class SolarCustom(DatetimeDataset):
    def __init__(self, solar_dataframe):
        super().__init__(target=solar_dataframe,
                         similarity_score='correntropy',
                         temporal_aggregation='sum',
                         spatial_aggregation='sum',
                         name='solar')

    def correntropy(self, x, period, mask=None, gamma=0.05):
        """Computes similarity matrix by looking at the similarity of windows of
        length `period` using correntropy.

        See Liu et al., "Correntropy: Properties and Applications in Non-Gaussian
        Signal Processing", TSP 2007.

        Args:
            x: Input series.
            period: Length of window.
            mask: Missing value mask.
            gamma: Width of the kernel

        Returns:
            The similarity matrix.
        """
        from sklearn.metrics.pairwise import rbf_kernel

        if mask is None:
            # mask = 1 - np.isnan(x, dtype='uint8')
            mask = 1 - np.isnan(x).astype('uint8')
            mask = mask[..., None]

        sim = np.zeros((x.shape[1], x.shape[1]))
        tot = np.zeros_like(sim)
        for i in range(period, len(x), period):
            xi = x[i - period:i].T
            m = mask[i - period:i].min(0)
            si = rbf_kernel(xi, gamma=gamma)
            m = m * m.T
            si = si * m
            sim += si
            tot += m
        return sim / (tot + tsl.epsilon)

    def compute_similarity(self,
                           method: str,
                           gamma=10,
                           trainlen=None,
                           **kwargs) -> Optional[np.ndarray]:
        train_df = self.dataframe()
        if trainlen is not None:
            train_df = self.dataframe().iloc[:trainlen]

        x = np.asarray(train_df)
        if method == 'correntropy':
            x = (x - x.mean()) / x.std()
            sim = self.correntropy(x, len(x), gamma=gamma)
        else:
            raise NotImplementedError
        return sim