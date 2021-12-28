import bioimageio.core
import h5py
import napari
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.core.prediction import predict_with_tiling
from xarray import DataArray


def predict_enhancer(model_path, pred_path):
    model = bioimageio.core.load_resource_description(model_path)
    with h5py.File(pred_path, "r") as f:
        rf_pred = f["exported_data"][..., 1]
        if "pred" in f:
            print("Load prediction from memory")
            return rf_pred, f["pred"][:]
    tiling = {"halo": {"x": 16, "y": 16, "z": 4},
              "tile": {"x": 96, "y": 96, "z": 8}}
    with create_prediction_pipeline(bioimageio_model=model) as pp:
        inputs = [DataArray(rf_pred[None, None], dims=tuple(pp.input_specs[0].axes))]
        pred = predict_with_tiling(pp, inputs, tiling, verbose=True)
    with h5py.File(pred_path, "a") as f:
        f.create_dataset("pred", data=pred, compression="gzip")
    return rf_pred, pred


def main():
    model = "./networks/cremi-v1.zip"
    raw_path = "./test_volumes/vol0.h5"
    pred_path = "./test_volumes/vol0-raw_Probabilities.h5"
    rf_pred, pred = predict_enhancer(model, pred_path)
    with h5py.File(raw_path, "r") as f:
        raw = f["raw"][:]
    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(rf_pred)
    v.add_image(pred)
    napari.run()


if __name__ == "__main__":
    main()
