import argparse
import napari
from cluster_tools.utils.volume_utils import load_mask
from elf.io import open_file

RESOLUTION = [0.04, 0.01, 0.01]
POSITIONS = [
    {"position": [96.68819639184461, 72.1844707668459, 48.2970615042884], "timepoint": 0},
    {"position": [90.40336311288618, 67.8636478875620, 45.4688865287571], "timepoint": 0},
    {"position": [88.36079229722469, 47.5950605629211, 50.2610719039629], "timepoint": 0},
    {"position": [92.99585684045653, 68.2564499674969, 45.0760844488222], "timepoint": 0}
]


# check the current segmentation and intermediates at one of the intersting points from jil
def check_full_seg(position, halo=[50, 512, 512]):
    raw_path = "../data/S016/images/bdv-n5/S016_aligned_full.n5"
    raw_key = "setup0/timepoint0/s0"
    bb = tuple(
        slice(int(pos - ha), int(pos + ha)) for pos, ha in zip(position, halo)
    )

    with open_file(raw_path, "r") as f:
        ds = f[raw_key]
        shape = ds.shape
        ds.n_threads = 8
        raw = f[bb]

    mask_path = "../data/S016/images/bdv-n5/foreground.n5"
    mask_key = "setup0/timepoint0/s0"
    mask_file = load_mask(mask_path, mask_key, shape)
    mask = mask_file[bb]

    tmp_path = "/scratch/pape/jils_project/full_seg/data.n5"
    rf_key = "predictions/rf"
    enh_key = "predictions/enhancer"
    with open_file(tmp_path, "r") as f:
        if rf_key in f:
            ds = f[rf_key]
            ds.n_threads = 8
            rf_pred = ds[bb]
        else:
            rf_pred = None
        if enh_key in f:
            ds = f[enh_key]
            ds.n_threads = 8
            enh = ds[:]
        else:
            enh = None

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(mask)
    if rf_pred is not None:
        v.add_image(rf_pred)
    if enh is not None:
        v.add_image(enh)
    napari.run()


def get_position(block_id):
    position = POSITIONS[block_id]
    position = [int(pos / res) for pos, res in zip(position[::-1], RESOLUTION)]
    return position


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_id", "-b", type=int, default=0)
    args = parser.parse_args()
    position = get_position(args.block_id)
    check_full_seg(position)


if __name__ == "__main__":
    main()
