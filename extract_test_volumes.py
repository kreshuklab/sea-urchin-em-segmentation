import os
import h5py
import z5py
from mobie.viewer_transformations import affine_to_position

DATA_PATH = "/g/emcf/ronchi/Lueter-Esther/S016_21-12-08_10nm_final/aligned/20211215/S016_aligned_full.n5"
KEY = "setup0/timepoint0/s0"

# FIXME Jil's positions with the nanometer project
# RESOLUTION = [40.0, 10.0, 10.0]
# these positions are out of bound in z! no clue why. Need to investigate.
# For now, just get the whole stack....
# POSITIONS = [
#     {"position": [155698.47319301794, 143414.10907393767, 85517.10851345764], "timepoint":0},
#     {"position": [193264.93622139096, 150122.3684566866, 97218.19676199825], "timepoint":0},
#     {"position": [193264.93622139096, 150122.3684566866, 93872.26549247361], "timepoint":0},
#     {"position": [168097.71406366222, 110553.09431274314, 93435.83967470954], "timepoint":0}
# ]

# test position
RESOLUTION = [0.04, 0.01, 0.01]
POSITIONS = [
    {"position": [85.89538020654878, 40.49100431959676, 41.49596211133668], "timepoint": 0}
]


def extract_test_volume(position, out_path, halo):
    # translate the position from physical to data coordinates,
    # note that java uses coordinates XYZ while python uses ZYX
    coordinate = [int(pos / res) for pos, res in zip(position[::-1], RESOLUTION)]
    print(coordinate, position[::-1])
    # the bounding box for cutting out the data
    bb = tuple(slice(coord - ha, coord + ha) for coord, ha in zip(coordinate, halo))
    with z5py.File(DATA_PATH, "r") as f:
        ds = f[KEY]
        print(ds.shape)
        data = ds[bb]
    # extract to hdf5, use e.g. imageio to extract to tif instead
    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=data, compression="gzip")


# extract all test volumes
def main():
    halo = [50, 512, 512]  # extract 100x1024x1024 pixels
    os.makedirs("./test_volumes", exist_ok=True)
    for vol_id, pos in enumerate(POSITIONS):
        out_path = f"./test_volumes/vol{vol_id}.h5"
        extract_test_volume(pos["position"], out_path, halo)


def aff():
    aff = [1.161857366456636, 0.0, 0.0, 2.3829614319540724,
           0.0, 1.161857366456636, 0.0, -95.49419071316771,
           0.0, 0.0, 1.161857366456636, -95.38848978608982]
    print(affine_to_position(aff))


if __name__ == "__main__":
    main()
    # aff()
