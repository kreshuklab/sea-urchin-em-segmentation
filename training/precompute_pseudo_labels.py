# precompute the pseudo labels
import argparse
import os
import subprocess

import bioimageio.core
from bioimageio.core.prediction import predict_with_tiling
from elf.io import open_file
from elf.wrapper.resized_volume import ResizedVolume
from ilastik.experimental.api import from_project_file
from xarray import DataArray


DATA_PATH = "/g/emcf/ronchi/Lueter-Esther/S016_21-12-08_10nm_final/aligned/20211215/S016_aligned_full.n5"
KEY = "setup0/timepoint0/s0"
RESOLUTION = [0.04, 0.01, 0.01]
POSITIONS = [
    {"position": [96.68819639184461, 72.1844707668459, 48.2970615042884], "timepoint": 0},
    {"position": [90.40336311288618, 67.8636478875620, 45.4688865287571], "timepoint": 0},
    {"position": [88.36079229722469, 47.5950605629211, 50.2610719039629], "timepoint": 0},
    {"position": [92.99585684045653, 68.2564499674969, 45.0760844488222], "timepoint": 0}
]


def _predict_autocontext(data, ilp, prefix):
    tmp_pred = f"./pseudo_labels/autocontext-tmp-{prefix}.h5"
    tmp_raw = f"./pseudo_labels/raw-tmp-{prefix}.h5"
    if os.path.exists(tmp_pred):
        os.remove(tmp_pred)

    with open_file(tmp_raw, "w") as f:
        f.create_dataset("raw", data=data, chunks=True)

    ilastik_folder = "/g/kreshuk/pape/Work/software/src/ilastik/ilastik-1.4.0b8-Linux"
    ilastik_exe = os.path.join(ilastik_folder, "run_ilastik.sh")
    assert os.path.exists(ilastik_exe), ilastik_exe
    input_str = f"{tmp_raw}/raw"
    cmd = [ilastik_exe, "--headless",
           "--project=%s" % ilp,
           "--output_format=compressed hdf5",
           "--raw_data=%s" % input_str,
           "--output_filename_format=%s" % tmp_pred,
           "--readonly=1"]
    print("Run ilastik prediction ...")
    subprocess.run(cmd)

    with open_file(tmp_pred, "r") as f:
        pred = f["exported_data"][:]
    return pred


def precompute_pseudo_labels_block(input_dataset, coordinate, output_path, ilp, enhancer, mask=None):
    # halo = [16, 32, 32]
    halo = [75, 768, 768]

    print("Load data ...")
    bb = tuple(slice(coord - ha, coord + ha) for coord, ha in zip(coordinate, halo))
    data = input_dataset[bb]

    if mask is not None:
        fg_mask = mask[bb].astype("bool")
        assert fg_mask.shape == data.shape

    # import napari
    # v = napari.Viewer()
    # v.add_image(data)
    # v.add_labels(fg_mask)
    # napari.run()
    # return

    # TODO support autocontext
    print("Predict ilastik ...")
    if isinstance(ilp, str):
        rf_pred = _predict_autocontext(data, ilp, prefix="vanilla" if mask is None else "mask")
    else:
        input_ = DataArray(data, dims=tuple("zyx"))
        rf_pred = ilp.predict(input_).values
    rf_pred = rf_pred[..., 1] - rf_pred[..., 0]

    print("Predict enhancer ...")
    rf_pred = DataArray(rf_pred[None, None], dims=tuple("bczyx"))
    tiling = {"halo": {"x": 32, "y": 32, "z": 4},
              "tile": {"x": 384, "y": 384, "z": 32}}
    pred = predict_with_tiling(enhancer, rf_pred, tiling=tiling, verbose=True)[0].values[0, 0]
    # pred = enhancer(rf_pred)[0].values[0, 0]

    if fg_mask is not None:
        pred[~fg_mask] = 0.0

    with open_file(output_path, "w") as f:
        f.create_dataset("raw", data=data, compression="gzip")
        f.create_dataset("pseudo-labels", data=pred, compression="gzip")


# pseudo labels smear out things a lot, cosinder using something much sparser, e.g. autocontext
def precompute_pseudo_labels(autocontext=False, apply_mask=False):
    root = "./pseudo_labels/autocontext" if autocontext else "./pseudo_labels/rf"
    out_dir = f"{root}/masked" if apply_mask else f"{root}/vanilla"
    os.makedirs(out_dir, exist_ok=True)

    model_path = "../networks/cremi-v1.zip"
    if autocontext:
        ilp = "../ilastik_projects/jil-3d-autocontext.ilp"
    else:
        ilp = from_project_file("../ilastik_projects/jil/vol3_3D_pixelclass.ilp")
    mask_path = "../segmentation/data/mask.h5"

    model = bioimageio.core.load_resource_description(model_path)
    with bioimageio.core.create_prediction_pipeline(bioimageio_model=model) as pp, open_file(DATA_PATH, "r") as f:
        ds = f[KEY]
        ds.n_threads = 8

        if apply_mask:
            with open_file(mask_path) as f:
                mask = f["neuropil"][:] > 0.5
            mask = ResizedVolume(mask, ds.shape, order=0)
        else:
            mask = None

        for block_id, pos in enumerate(POSITIONS):
            output_path = os.path.join(out_dir, f"block-{block_id}.h5")
            print("Precomputing pseudo label block", block_id)
            # if os.path.exists(output_path):
            #     print("already exists")
            #     continue
            coordinate = [int(po / res) for po, res in zip(pos["position"][::-1], RESOLUTION)]
            precompute_pseudo_labels_block(ds, coordinate, output_path, ilp, pp, mask=mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--autocontext", default=0, type=int)
    parser.add_argument("-m", "--apply_mask", default=0, type=int)
    args = parser.parse_args()
    precompute_pseudo_labels(bool(args.autocontext), bool(args.apply_mask))


if __name__ == "__main__":
    main()