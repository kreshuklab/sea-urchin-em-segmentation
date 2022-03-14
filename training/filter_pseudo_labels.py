import os
from glob import glob

import bioimageio.core
import vigra
from bioimageio.core.prediction import predict_with_padding
from elf.io import open_file
from ilastik.experimental.api import from_project_file
from scipy.ndimage.morphology import binary_erosion
from xarray import DataArray

DATA_PATH = "../data/S016/images/bdv-n5/S016_aligned_full.n5"
POSITIONS = [
    {"position": [96.68819639184461, 72.1844707668459, 48.2970615042884], "timepoint": 0},
    {"position": [90.40336311288618, 67.8636478875620, 45.4688865287571], "timepoint": 0},
    {"position": [88.36079229722469, 47.5950605629211, 50.2610719039629], "timepoint": 0},
    {"position": [92.99585684045653, 68.2564499674969, 45.0760844488222], "timepoint": 0}
]


def _predict(block_id, shape):
    path = "./pseudo_labels/tmp_nuclei"
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f"{block_id}.h5")
    if os.path.exists(path):
        with open_file(path, "r") as f:
            return f["raw"][:], f["pred"][:]

    resolution = [0.08, 0.08, 0.08]
    position = POSITIONS[block_id]["position"]
    coordinate = [int(pos / res) for pos, res in zip(position[::-1], resolution)]

    scale_factor = [2, 8, 8]
    halo = [int(sh // sf // 2) for sh, sf in zip(shape, scale_factor)]
    bb = tuple(slice(coord - ha, coord + ha) for coord, ha in zip(coordinate, halo))
    with open_file(DATA_PATH, "r") as f:
        ds = f["setup0/timepoint0/s3"]
        data = ds[bb]

    ilp = from_project_file("../ilastik_projects/jil-nucleus-seg.ilp")
    input_ilp = DataArray(data, dims=tuple("zyx"))
    pred_ilp = ilp.predict(input_ilp).values
    pred_ilp = pred_ilp[..., 1] - pred_ilp[..., 0]

    model_path = "./enhancer_training/networks/nuclei-v1/nuclei-v1.zip"
    input_ = DataArray(pred_ilp[None, None], dims=tuple("bczyx"))
    with bioimageio.core.create_prediction_pipeline(
        bioimageio_model=bioimageio.core.load_resource_description(model_path)
    ) as pp:
        pred = predict_with_padding(pp, input_, padding=True)[0].values.squeeze()

    with open_file(path, "w") as f:
        f.create_dataset("raw", data=data, compression="gzip")
        f.create_dataset("pred", data=pred, compression="gzip")

    return data, pred


def get_nucleus_mask(block_id, shape):
    raw, pred = _predict(block_id, shape)
    threshold = 0.5
    mask = pred > threshold
    mask = binary_erosion(mask, iterations=2)

    # import napari
    # v = napari.Viewer()
    # v.add_image(raw)
    # v.add_image(pred)
    # v.add_labels(mask)
    # napari.run()

    mask = vigra.sampling.resize(mask.astype("float32"), shape, order=0).astype("bool")
    assert mask.shape == shape
    return mask


def filter_nuclei(block_id, input_path, output_path):
    with open_file(input_path, "r") as f:
        raw = f["raw"][:]
        labels = f["pseudo-labels"][:]
    nucleus_mask = get_nucleus_mask(block_id, labels.shape)
    labels[nucleus_mask] = 0.0
    with open_file(output_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip", chunks=(25, 256, 256))
        f.create_dataset("pseudo-labels", data=labels, compression="gzip", chunks=(25, 256, 256))


def get_carving_mask(block_id, shape):
    mask_path = f"./pseudo_labels/carving-masks/carving-block{block_id}.h5"
    print("Load mask")
    with open_file(mask_path, "r") as f:
        mask = f["exported_data"][:].squeeze() != 0
    # print("Filter mask")
    # for z in range(mask.shape[0]):
    #     mask[z] = binary_erosion(mask[z], iterations=2)
    print("Resize mask")
    mask = vigra.sampling.resize(mask.astype("float32"), shape, order=0).astype("bool")
    assert mask.shape == shape
    return mask


def filter_carving(block_id, input_, output_path):
    with open_file(input_, "r") as f:
        raw = f["raw"][:]
        labels = f["pseudo-labels"][:].squeeze()
    carving_mask = get_carving_mask(block_id, labels.shape)
    labels[carving_mask] = 0.0
    with open_file(output_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip", chunks=(25, 256, 256))
        f.create_dataset("pseudo-labels", data=labels, compression="gzip", chunks=(25, 256, 256))


def filter_all_pseudo_labels(root):
    input_data = glob(os.path.join(root, "vanilla", "*.h5"))
    output_folder = os.path.join(root, "masked-carving")
    os.makedirs(output_folder, exist_ok=True)
    for block_id, input_ in enumerate(input_data):
        print("Filter labels for", input_)
        output_path = os.path.join(output_folder, os.path.basename(input_))
        # filter_nuclei(block_id, input_, output_path)
        filter_carving(block_id, input_, output_path)


if __name__ == "__main__":
    # filter_all_pseudo_labels("./pseudo_labels/rf")
    # filter_all_pseudo_labels("./pseudo_labels/autocontext")
    # filter_all_pseudo_labels("./pseudo_labels/rf2")
    filter_all_pseudo_labels("./pseudo_labels/segmentor")
