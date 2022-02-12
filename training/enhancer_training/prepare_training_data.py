import os
from glob import glob
from elf.io import open_file
from skimage.transform import downscale_local_mean, resize


def prepare_volume(in_path, out_path, raw_key="volumes/raw", label_key="volumes/labels/neuron_ids"):
    with open_file(in_path, "r") as f:
        raw = f[raw_key][:]
        gt = f[label_key][:]
    assert raw.shape == gt.shape
    raw = downscale_local_mean(raw, (1, 2, 2))
    gt = resize(gt, raw.shape, order=0, preserve_range=True, anti_aliasing=False).astype(gt.dtype)
    with open_file(out_path, "a") as f:
        f.create_dataset(raw_key, data=raw, compression="gzip")
        f.create_dataset(label_key, data=gt, compression="gzip")


def prepare_cremi():
    os.makedirs("./data/neurons", exist_ok=True)
    in_paths = [
        "/g/kreshuk/data/cremi/original/sample_A_20160501.hdf",
        "/g/kreshuk/data/cremi/original/sample_B_20160501.hdf",
        "/g/kreshuk/data/cremi/original/sample_C_20160501.hdf",
    ]
    out_paths = [
        "./data/neurons/sampleA.h5",
        "./data/neurons/sampleB.h5",
        "./data/neurons/sampleC.h5",
    ]
    for path, out_path in zip(in_paths, out_paths):
        prepare_volume(path, out_path)


def prepare_platy_cells():
    in_paths = glob("/scratch/pape/platy/membrane/*.n5")
    out_dir = "./data/cells"
    os.makedirs(out_dir)
    for path in in_paths:
        out_path = os.path.join(out_dir, os.path.basename(path))
        prepare_volume(path, out_path, "volumes/raw/s1", "volumes/labels/segmentation/s1")


if __name__ == "__main__":
    # prepare_cremi()
    prepare_platy_cells()
