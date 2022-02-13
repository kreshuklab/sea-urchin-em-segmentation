import os
from glob import glob

import numpy as np
from elf.io import open_file
from elf.wrapper.resized_volume import ResizedVolume
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


def prepare_neuropil_data():
    data_path = "/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/sbem-6dpf-1-whole-raw.n5"
    data_key = "setup0/timepoint0/s3"
    with open_file(data_path, "r") as f:
        ds = f[data_key]
        shape = ds.shape
    print(shape)

    lab_path = "/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/sbem-6dpf-1-whole-segmented-neuropil.n5"
    lab_key = "setup0/timepoint0/s0"
    with open_file(lab_path, "r") as f:
        ds = f[lab_key]
        ds.n_threads = 16
        print("Loading data ...")
        labels = ds[:]
    print("Resize labels ...")
    labels = ResizedVolume(labels, shape, order=0)[:]
    assert labels.shape == shape
    print(labels.min(), labels.max())

    print("Find bounding box ...")
    mask = np.where(labels > 0)
    bb = tuple(slice(
        int(np.min(ma)), int(np.max(ma)) + 1
    ) for ma in mask)
    print(bb)
    labels = labels[bb]

    print("Load raw data ...")
    with open_file(data_path, "r") as f:
        ds = f[data_key]
        ds.n_threads = 16
        raw = ds[bb]
    assert raw.shape == labels.shape

    save_path = "./data/neuropil/data.n5"
    os.makedirs("./data/neuropil", exist_ok=True)
    print("Save data ...")
    with open_file(save_path, "a") as f:
        f.create_dataset("raw", data=raw, compression="gzip", chunks=(96, 96, 96))
        f.create_dataset("labels", data=labels, compression="gzip", chunks=(96, 96, 96))


if __name__ == "__main__":
    # prepare_cremi()
    # prepare_platy_cells()
    prepare_neuropil_data()
