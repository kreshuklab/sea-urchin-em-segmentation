import argparse
import os

import bioimageio.core
import elf.segmentation as eseg
import h5py

from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.core.prediction import predict_with_tiling
from elf.segmentation.utils import sharpen_edges
from ilastik.experimental.api import from_project_file
from xarray import DataArray


def predict_rf(raw, path, ilp, force=False):
    with h5py.File(path, "r") as f:
        if "rf_pred" in f and not force:
            print("Load random forest prediction ...")
            return f["rf_pred"][:]
    print("Run random forest prediction ...")
    ilp = from_project_file(ilp)
    input_ = DataArray(raw, dims=("z", "y", "x"))
    pred = ilp.predict(input_).values[..., 1]
    with h5py.File(path, "a") as f:
        ds = f.require_dataset("rf_pred", shape=pred.shape, compression="gzip", dtype=pred.dtype)
        ds[:] = pred
    return pred


def predict_enhancer(rf_pred, path, model, force=False):
    model = bioimageio.core.load_resource_description(model)
    with h5py.File(path, "r") as f:
        if "enhancer_pred" in f and not force:
            print("Load prediction from memory ...")
            return f["enhancer_pred"][:]
    print("Run enhancer prediction ...")
    tiling = {"halo": {"x": 16, "y": 16, "z": 4},
              "tile": {"x": 224, "y": 224, "z": 12}}
    with create_prediction_pipeline(bioimageio_model=model) as pp:
        inputs = [DataArray(rf_pred[None, None], dims=tuple(pp.input_specs[0].axes))]
        pred = predict_with_tiling(pp, inputs, tiling, verbose=True)[0].values.squeeze()
    with h5py.File(path, "a") as f:
        ds = f.require_dataset("enhancer_pred", shape=pred.shape, compression="gzip", dtype=pred.dtype)
        ds[:] = pred
    return pred


def segment(pred, path, force=False, n_threads=8):
    with h5py.File(path, "r") as f:
        if "segmentation" in f and not force:
            print("Load segmentation from memory ...")
            return f["segmentation"][:]
    print("Run multicut segmentation ...")
    ws, max_id = eseg.stacked_watershed(pred, n_threads=n_threads, threshold=0.3, sigma_seeds=2.0)
    rag = eseg.compute_rag(ws, max_id + 1, n_threads)
    mean_and_len = eseg.compute_boundary_mean_and_length(rag, pred, n_threads)
    costs, lens = mean_and_len[:, 0], mean_and_len[:, 1]

    # TODO check if sharpening the costs helps and/or if weighting schemes help
    weight_scheme = None
    costs = sharpen_edges(costs)
    costs = eseg.compute_edge_costs(costs, edge_sizes=lens, weighting_scheme=weight_scheme)

    node_labels = eseg.multicut.multicut_decomposition(rag, costs, n_threads=n_threads)
    seg = eseg.project_node_labels_to_pixels(rag, node_labels, n_threads)
    with h5py.File(path, "a") as f:
        ds = f.require_dataset("segmentation", shape=seg.shape, compression="gzip", dtype=seg.dtype)
        ds[:] = seg
    return seg


def view_segmentation(raw, rf_pred, pred, seg):
    import napari
    print("start viewer")
    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(rf_pred, name="random forest")
    v.add_image(pred, name="enhancer")
    v.add_labels(seg, name="segmentation")
    napari.run()


def segment_test_volume(path, ilp, model, force, view):
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
    rf_pred = predict_rf(raw, path, ilp, force)
    pred = predict_enhancer(rf_pred, path, model, force)
    seg = segment(pred, path, force)
    if view:
        view_segmentation(raw, rf_pred, pred, seg)


def main():
    this_folder = os.path.split(__file__)[0]
    default_ilp = os.path.join(this_folder, "../ilastik_projects/boundary_prediction_v2.ilp")
    default_model = os.path.join(this_folder, "../networks/cremi-v1.zip")

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vol_id", type=int, required=True)
    parser.add_argument("-i", "--ilp", default=default_ilp)
    parser.add_argument("-m", "--model", default=default_model)
    parser.add_argument("-f", "--force", type=int, default=0)
    parser.add_argument("--view", type=int, default=0)
    args = parser.parse_args()
    path = os.path.join(this_folder, f"../test_volumes/vol{args.vol_id}.h5")
    segment_test_volume(path, args.ilp, args.model, bool(args.force), bool(args.view))


if __name__ == "__main__":
    main()
