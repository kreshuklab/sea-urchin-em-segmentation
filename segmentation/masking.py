import os
import h5py
import mobie
import z5py
from ilastik.experimental.api import from_project_file
from xarray import DataArray


def compute_masks(raw_path, ilp_path):
    scale = 4
    with z5py.File(raw_path, "r") as f:
        raw = f[f"setup0/timepoint0/s{scale}"][:]
        print("shape:", raw.shape)

    out_path = "data/mask.h5"
    if os.path.exists(out_path):
        with h5py.File(out_path, "r") as f:
            bg = f["bg"][:]
    else:
        ilp = from_project_file(ilp_path)
        input_ = DataArray(raw, dims=("z", "y", "x"))
        pred = ilp.predict(input_).values
        neuropil, nuclei, bg = pred[..., 0], pred[..., 1], pred[..., 2]
        with h5py.File(out_path, "a") as f:
            f.create_dataset("neuropil", data=neuropil, compression="gzip")
            f.create_dataset("nuclei", data=nuclei, compression="gzip")
            f.create_dataset("bg", data=bg, compression="gzip")
    return bg


def make_fg_mask(bg):
    fg = 1. - bg
    fg_mask = (fg > 0.5).astype("uint8")
    out_path = "data/fg_mask.h5"
    with h5py.File(out_path, "a") as f:
        f.create_dataset("data", data=fg_mask, compression="gzip")

    # the mask is at s4
    resolution = []  # TODO
    scale_factors = []
    chunks = []

    mobie.add_image(out_path, "data",
                    root="../data", dataset_name="S016",
                    image_name="foreground", resolution=resolution,
                    scale_factors=scale_factors, chunks=chunks,
                    menu_name="mask")


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
    bg = compute_masks(raw_path, ilp_path)
    # make_fg_mask(bg)


if __name__ == "__main__":
    main()
