import os
import pickle

import h5py
import numpy as np
import vigra

from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

FILTERS = [vigra.filters.gaussianSmoothing,
           vigra.filters.laplacianOfGaussian,
           vigra.filters.gaussianGradientMagnitude,
           vigra.filters.hessianOfGaussianEigenvalues,
           vigra.filters.structureTensorEigenvalues]
SIGMAS = [
    (0.25, 1.0, 1.0),
    (0.6, 2.4, 2.4),
    (1.0, 4.0, 4.0),
    (2.0, 8.0, 8.0)
]


def train_anisotropic_rf():
    save_path = "./data/anisotropic_rf.pkl"
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            rf = pickle.load(f)
        return rf

    raw_path = "../test_volumes/vol3.h5"
    with h5py.File(raw_path, "r") as f:
        raw = f["raw"][:].astype("float32")
    label_path = "../ilastik_projects/jil/pixellabels_vol3_3D.h5"
    with h5py.File(label_path, "r") as f:
        labels = f["exported_data"][..., 0]

    rf = RandomForestClassifier(n_estimators=10, n_jobs=16)
    features = []

    label_mask = labels > 0
    print("Computing features...")
    for filter_id, filter_ in tqdm(enumerate(FILTERS), total=len(FILTERS)):
        for sigma in SIGMAS:
            if filter_id == len(FILTERS) - 1:
                outer_scale = tuple(2*sig for sig in sigma)
                response = filter_(raw, sigma, outer_scale)
            else:
                response = filter_(raw, sigma)

            if response.ndim == 3:
                features.append(response[label_mask][:, None])
            else:
                for c in range(response.shape[-1]):
                    features.append(response[..., c][label_mask][:, None])

    features = np.concatenate(features, axis=1)
    labels = labels[label_mask]
    assert len(features) == len(labels)
    labels -= 1
    assert (np.unique(labels) == np.array([0, 1])).all()
    print(features.shape)
    print(labels.shape)

    rf.fit(features, labels)
    with open(save_path, "wb") as f:
        pickle.dump(rf, f)
    return rf


def predict_anisotropic_rf(rf, block_id=0):
    path = f"../test_volumes/vol{block_id}.h5"
    with h5py.File(path, "r") as f:
        raw = f["raw"][:].astype("float32")

    print("Computing features...")
    features = []
    for filter_id, filter_ in tqdm(enumerate(FILTERS), total=len(FILTERS)):
        for sigma in SIGMAS:
            if filter_id == len(FILTERS) - 1:
                outer_scale = tuple(2*sig for sig in sigma)
                response = filter_(raw, sigma, outer_scale)
            else:
                response = filter_(raw, sigma)

            if response.ndim == 3:
                features.append(response.flatten()[:, None])
            else:
                for c in range(response.shape[-1]):
                    features.append(response[..., c].flatten()[:, None])
    features = np.concatenate(features, axis=1)
    print(features.shape)

    print("Predicting probabilities ...")
    probs = rf.predict_proba(features)[:, 1]
    probs = probs.reshape(raw.shape)

    print("Saving probabilities ...")
    out_path = f"./data/vol{block_id}.h5"
    with h5py.File(out_path, "a") as f:
        f.create_dataset("anisotropic_rf/rf_pred", data=probs, compression="gzip")


def main():
    rf = train_anisotropic_rf()
    predict_anisotropic_rf(rf)


if __name__ == "__main__":
    main()
