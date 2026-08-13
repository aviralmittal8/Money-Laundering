import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler


class HasherPreprocessor:
    def __init__(self, numeric_features, categorical_features, hash_dim=128):
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.hash_dim = int(hash_dim)
        self.scaler = StandardScaler()
        self.hasher = FeatureHasher(n_features=self.hash_dim, input_type="string")

    def _cat_rows(self, df):
        rows = []
        for _, row in df[self.categorical_features].iterrows():
            rows.append([f"{col}={row[col]}" for col in self.categorical_features])
        return rows

    def fit_transform(self, df):
        num = df[self.numeric_features].astype(float).to_numpy()
        num_scaled = self.scaler.fit_transform(num)
        cat_hashed = self.hasher.transform(self._cat_rows(df)).toarray().astype(np.float32)
        return np.hstack([num_scaled.astype(np.float32), cat_hashed]).astype(np.float32)

    def transform(self, df):
        num = df[self.numeric_features].astype(float).to_numpy()
        num_scaled = self.scaler.transform(num)
        cat_hashed = self.hasher.transform(self._cat_rows(df)).toarray().astype(np.float32)
        return np.hstack([num_scaled.astype(np.float32), cat_hashed]).astype(np.float32)
