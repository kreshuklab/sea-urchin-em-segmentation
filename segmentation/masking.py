import os
import h5py
import mobie
import z5py

from ilastik.experimental.api import from_project_file
from pybdv.metadata import get_resolution
from pybdv.util import get_scale_factors
from xarray import DataArray


def compute_masks(raw_path, ilp_path):
    scale = 4

    out_path = "data/mask.h5"
    if os.path.exists(out_path):
        return out_path

    with z5py.File(raw_path, "r") as f:
        raw = f[f"setup0/timepoint0/s{scale}"][:]
        print("shape:", raw.shape)

    ilp = from_project_file(ilp_path)
    input_ = DataArray(raw, dims=("z", "y", "x"))
    pred = ilp.predict(input_).values
    neuropil, nuclei, bg = pred[..., 0], pred[..., 1], pred[..., 2]
    with h5py.File(out_path, "a") as f:
        f.create_dataset("neuropil", data=neuropil, compression="gzip")
        f.create_dataset("nuclei", data=nuclei, compression="gzip")
        f.create_dataset("bg", data=bg, compression="gzip")

    return out_path


def make_fg_mask(mask_path):
    out_path = "data/fg_mask.h5"
    if os.path.exists(out_path):
        return out_path
    with h5py.File(mask_path, "r") as f:
        bg = f["bg"][:]
    fg = 1. - bg
    fg_mask = (fg > 0.5).astype("uint8")
    with h5py.File(out_path, "a") as f:
        f.create_dataset("data", data=fg_mask, compression="gzip")
    return out_path


def add_to_mobie(mask_path, raw_path):
    xml_path = os.path.join(os.path.split(raw_path)[0], "raw.xml")
    # the mask is at s4
    scale = 4
    full_resolution = get_resolution(xml_path, setup_id=0)
    scale_factors = get_scale_factors(raw_path, setup_id=0)
    scale_factor = scale_factors[scale]
    resolution = [res * sf for res, sf in zip(full_resolution, scale_factor)]
    scale_factors = 3 * [[2, 2, 2]]
    chunks = [32, 128, 128]
    view = mobie.metadata.get_default_view("image", "foreground", "mask", contrastLimits=[0, 1])
    mobie.add_image(mask_path, "data",
                    root="../data", dataset_name="S016",
                    image_name="foreground", resolution=resolution,
                    scale_factors=scale_factors, chunks=chunks,
                    menu_name="mask", view=view, max_jobs=16)


# - predict the different masks (neuropil, nuclei, resin)
# - binarize the predictions
# - add to mobie
def main():
    this_folder = os.path.split(__file__)[0]
    raw_path = os.path.join(
        this_folder, "../data/S016/images/bdv-n5/S016_aligned_full.n5"
    )
    ilp_path = os.path.join(
        this_folder, "../ilastik_projects/jil/masking_im_s4downsampled.ilp"
    )
    mask_path = compute_masks(raw_path, ilp_path)
    mask_path = make_fg_mask(mask_path)
    add_to_mobie(mask_path, raw_path)


if __name__ == "__main__":
    main()
