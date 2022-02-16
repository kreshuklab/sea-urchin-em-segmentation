import os
import h5py
import z5py

DATA_PATH = "./data/S016/images/bdv-n5/S016_aligned_full.n5"
POSITIONS = [
    {"position": [96.68819639184461, 72.1844707668459, 48.2970615042884], "timepoint": 0},
    {"position": [90.40336311288618, 67.8636478875620, 45.4688865287571], "timepoint": 0},
    {"position": [88.36079229722469, 47.5950605629211, 50.2610719039629], "timepoint": 0},
    {"position": [92.99585684045653, 68.2564499674969, 45.0760844488222], "timepoint": 0}
]
RESOLUTIONS = [
    [0.04, 0.01, 0.01],
    [0.04, 0.02, 0.02],
    [0.04, 0.04, 0.04],
    [0.08, 0.08, 0.08],
]


def extract_test_volume(position, scale, out_path, halo):
    key = f"setup0/timepoint0/s{scale}"
    resolution = RESOLUTIONS[scale]
    # translate the position from physical to data coordinates,
    # note that java uses coordinates XYZ while python uses ZYX
    coordinate = [int(pos / res) for pos, res in zip(position[::-1], resolution)]
    print(coordinate, position[::-1])
    # the bounding box for cutting out the data
    bb = tuple(slice(coord - ha, coord + ha) for coord, ha in zip(coordinate, halo))
    with z5py.File(DATA_PATH, "r") as f:
        ds = f[key]
        print(ds.shape)
        data = ds[bb]
    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=data, compression="gzip")


# extract all test volumes
def main():
    scale = 3
    halo = [25, 256, 256]  # extract 50x512x512 pixels
    os.makedirs(f"./test_volumes/s{scale}", exist_ok=True)
    for vol_id, pos in enumerate(POSITIONS):
        out_path = f"./test_volumes/s{scale}/vol{vol_id}.h5"
        extract_test_volume(pos["position"], scale, out_path, halo)


if __name__ == "__main__":
    main()
