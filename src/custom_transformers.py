# this file is a module containing transformers compatible with sklearn’s API 
# (they implement .fit() and .transform()), since each class inherits
# from sklearn's BaseEstimator and TransformerMixin

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd

direction_map = {
"N": 0.0,
"NNE": 22.5,
"NE": 45.0,
"ENE": 67.5,
"E": 90.0,
"ESE": 112.5,
"SE": 135.0,
"SSE": 157.5,
"S": 180.0,
"SSW": 202.5,
"SW": 225.0,
"WSW": 247.5,
"W": 270.0,
"WNW": 292.5,
"NW": 315.0,
"NNW": 337.5
}

# custom transformer to convert direction labels to numeric values 
class DirectionEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.direction_map = direction_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # working with sin and cos to do the linear interpolation later
        angle = X["Direction"].map(self.direction_map)
        X["Direction_sin"] = np.sin(np.deg2rad(angle))
        X["Direction_cos"] = np.cos(np.deg2rad(angle))

        return X.drop(columns=["Direction"])


class Interpolator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["Speed"] = X["Speed"].interpolate(method="time")
        X["Direction_sin"] = X["Direction_sin"].interpolate(method="time")
        X["Direction_cos"] = X["Direction_cos"].interpolate(method="time")

        return X