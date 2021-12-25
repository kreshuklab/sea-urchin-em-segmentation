import os
import h5py
from skimage.transform import downscale_local_mean, resize


def prepare_volume(in_path, out_path):
    with h5py.File(in_path, "r") as f:
        raw = f["volumes/raw"][:]
        gt = f["volumes/labels/neuron_ids"][:]
    raw = downscale_local_mean(raw, (1, 2, 2))
    gt = resize(gt, raw.shape, order=0, preserve_range=True, anti_aliasing=False).astype(gt.dtype)
    with h5py.File(out_path, "a") as f:
        f.create_dataset("volumes/raw", data=raw, compression="gzip")
        f.create_dataset("volumes/labels/neuron_ids", data=gt, compression="gzip")


def prepare_all():
    os.makedirs("./data", exist_ok=True)
    in_paths = [
        "/home/pape/Work/data/cremi/sample_A_20160501.hdf",
    ]
    out_paths = [
        "./data/sampleA.h5",
    ]
    for path, out_path in zip(in_paths, out_paths):
        prepare_volume(path, out_path)


if __name__ == "__main__":
    prepare_all()
