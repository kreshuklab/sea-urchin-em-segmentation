import os
import argparse

import h5py
import napari


def compare_test_results(path, res_path):
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]

    results = {}
    with h5py.File(res_path, "r") as f:
        for prefix, g in f.items():
            for name, ds in g.items():
                print(prefix, name, ds.shape)
                results[f"{prefix}/{name}"] = ds[:]

    v = napari.Viewer()
    v.add_image(raw)
    for name, data in results.items():
        if "segmentation" in name:
            v.add_labels(data, name=name)
        else:
            v.add_image(data, name=name)
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vol_id", type=int, required=True)
    args = parser.parse_args()
    this_folder = os.path.split(__file__)[0]
    path = os.path.join(this_folder, f"../test_volumes/vol{args.vol_id}.h5")
    res_path = f"./data/vol{args.vol_id}.h5"
    compare_test_results(path, res_path)


if __name__ == "__main__":
    main()
