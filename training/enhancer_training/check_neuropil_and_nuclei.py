import argparse
import os

import bioimageio.core
import h5py
import napari
from bioimageio.core.prediction import predict_with_tiling
from ilastik.experimental.api import from_project_file
from xarray import DataArray


def predict_nuclei(data, out_path):
    if os.path.exists(out_path):
        with h5py.File(out_path, "r") as f:
            if "nuc_ilp" in f:
                assert "nuc" in f
                return f["nuc_ilp"][:], f["nuc"][:]

    ilp = from_project_file("../../ilastik_projects/jil-nucleus-seg.ilp")

    input_ilp = DataArray(data, dims=tuple("zyx"))
    print("Predict ilastik...")
    nuclei_ilp = ilp.predict(input_ilp).values
    nuclei_ilp = nuclei_ilp[..., 1] - nuclei_ilp[..., 0]

    print("Predict nuclei...")
    tiling = {"tile": {"x": 256, "y": 256, "z": 32}, "halo": {"x": 16, "y": 16, "z": 4}}
    input_nuc = DataArray(nuclei_ilp[None, None], dims=tuple("bczyx"))
    with bioimageio.core.create_prediction_pipeline(
        bioimageio_model=bioimageio.core.load_resource_description("./networks/nuclei-v1/nuclei-v1.zip")
    ) as pp:
        nuc = predict_with_tiling(pp, input_nuc, tiling=tiling)[0].values.squeeze()

    with h5py.File(out_path, "a") as f:
        f.create_dataset("nuc", data=nuc, compression="gzip")
        f.create_dataset("nuc_ilp", data=nuclei_ilp, compression="gzip")
    return nuclei_ilp, nuc


def predict_neuropil(data, out_path):
    with h5py.File(out_path, "r") as f:
        if "npil" in f:
            return f["npil"][:]
    print("Predict neuropil...")
    input_npil = DataArray(data[None, None], dims=tuple("bczyx"))
    tiling = {"tile": {"x": 256, "y": 256, "z": 32}, "halo": {"x": 16, "y": 16, "z": 4}}
    with bioimageio.core.create_prediction_pipeline(
        bioimageio_model=bioimageio.core.load_resource_description("./networks/neuropil-v1/neuropil-v1.zip")
    ) as pp:
        npil = predict_with_tiling(pp, input_npil, tiling=tiling)[0].values.squeeze()
    with h5py.File(out_path, "a") as f:
        f.create_dataset("npil", data=npil, compression="gzip")
    return npil


def check_neuropil_and_nuclei(vol_id):
    with h5py.File(f"../../test_volumes/s3/vol{vol_id}.h5", "r") as f:
        data = f["raw"][:]

    out_path = f"./data/predictions/nuc_npil{vol_id}.h5"
    os.makedirs("./data/predictions", exist_ok=True)
    nuclei_ilp, nuc = predict_nuclei(data, out_path)
    # npil = predict_neuropil(data, out_path)

    v = napari.Viewer()
    v.add_image(data)
    v.add_image(nuclei_ilp)
    v.add_image(nuc)
    # v.add_image(npil)
    napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vol_id", type=int, default=0)
    args = parser.parse_args()
    check_neuropil_and_nuclei(args.vol_id)
