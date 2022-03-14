import os
from glob import glob
import napari
import h5py


def check_pre(path):
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        labels = f["pseudo-labels"][:]
    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(labels)
    v.title = os.path.basename(path)
    napari.run()


def check_precomputed(input_):
    files = glob(os.path.join(input_, "*.h5"))
    files.sort()
    for ff in files:
        check_pre(ff)


# check_precomputed("./pseudo_labels/rf/masked-carving")
# check_precomputed("pseudo_labels/rf2/masked-carving")
check_precomputed("pseudo_labels/segmentor/masked-carving")
