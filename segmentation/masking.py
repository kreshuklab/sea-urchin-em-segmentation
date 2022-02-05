# - predict the different masks (neuropil, nuclei, resin)
# - binarize the predictions
# - add to mobie

import os

import napari
import h5py
import z5py
from ilastik.experimental.api import from_project_file
from xarray import DataArray


def compute_masks(raw_path, ilp_path):
    scale = 4
    with z5py.File(raw_path, "r") as f:
        raw = f[f"setup0/timepoint0/s{scale}"][:]
        print("shape:", raw.shape)

    tmp_path = "mask_tmp.h5"
    if os.path.exists(tmp_path):
        with h5py.File(tmp_path, "r") as f:
            pred = f["data"]
    else:
        ilp = from_project_file(ilp_path)
        input_ = DataArray(raw, dims=("z", "y", "x"))
        pred = ilp.predict(input_)[0].values
        with h5py.File(tmp_path, "a") as f:
            f.create_dataset("data", data=pred, compression="gzip")

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(pred)
    napari.run()


def main():
    this_folder = os.path.split(__file__)[0]
    raw_path = os.path.join(
        this_folder, "../data/S016/images/bdv-n5/S016_aligned_full.n5"
    )
    ilp_path = os.path.join(
        this_folder, "../ilastik_projects/jil/masking_im_s4downsampled.ilp"
    )
    compute_masks(raw_path, ilp_path)


if __name__ == "__main__":
    main()
