import os
from glob import glob

import h5py
import vigra


def prepare_for_carving(input_path, output_path, scale_factor=(1, 4, 4)):
    with h5py.File(input_path, "r") as f:
        raw = f["raw"][:].astype("float32")
    new_shape = tuple(sh // sf for sh, sf in zip(raw.shape, scale_factor))
    print("scale to shape", new_shape)
    raw = vigra.sampling.resize(raw, new_shape).clip(0, 255).astype("uint8")
    with h5py.File(output_path, "a") as f:
        f.create_dataset("raw", data=raw, compression="gzip")


def prepare_all():
    inputs = glob("./pseudo_labels/rf/vanilla/*.h5")
    out_dir = "./pseudo_labels/carving"
    os.makedirs(out_dir, exist_ok=True)
    for input_ in inputs:
        print(input_)
        out_path = os.path.join(out_dir, os.path.basename(input_))
        prepare_for_carving(input_, out_path)


prepare_all()
