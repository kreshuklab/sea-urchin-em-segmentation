import argparse

import bioimageio.core
import h5py
import napari

from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.core.prediction import predict_with_tiling
from ilastik.experimental.api import from_project_file
from xarray import DataArray


def predict_rf(ilp, raw, path):
    with h5py.File(path, "r") as f:
        if "rf_pred" in f:
            print("Load random forest prediction ...")
            return f["rf_pred"][:]
    print("Run random forest prediction ...")
    ilp = from_project_file(ilp)
    input_ = DataArray(raw, dims=("z", "y", "x"))
    pred = ilp.predict(input_).values
    with h5py.File(path, "a") as f:
        f.create_dataset("rf_pred", data=pred, compression="gzip")
    return pred


def predict_enhancer(model_path, rf_pred, path):
    model = bioimageio.core.load_resource_description(model_path)
    with h5py.File(path, "r") as f:
        if "enhancer_pred" in f:
            print("Load prediction from memory ...")
            return f["enhancer_pred"][:]
    print("Run enhancer prediction ...")
    tiling = {"halo": {"x": 16, "y": 16, "z": 4},
              "tile": {"x": 96, "y": 96, "z": 8}}
    with create_prediction_pipeline(bioimageio_model=model) as pp:
        inputs = [DataArray(rf_pred[None, None], dims=tuple(pp.input_specs[0].axes))]
        pred = predict_with_tiling(pp, inputs, tiling, verbose=True)
    with h5py.File(path, "a") as f:
        f.create_dataset("enhancer_pred", data=pred, compression="gzip")
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vol_id", type=int, default=0)
    args = parser.parse_args()
    vol_id = args.vol_id

    ilp = "./boundary_prediction.ilp"
    model = "./networks/cremi-v1.zip"
    raw_path = f"./test_volumes/vol{vol_id}.h5"
    with h5py.File(raw_path, "r") as f:
        raw = f["raw"][:]

    rf_pred = predict_rf(ilp, raw, raw_path)
    pred = predict_enhancer(model, rf_pred[..., 1], raw_path)

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(rf_pred[..., 1])
    v.add_image(pred.squeeze())
    napari.run()


if __name__ == "__main__":
    main()
