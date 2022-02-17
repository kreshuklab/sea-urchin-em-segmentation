import argparse
import json
import os

import imageio
import zarr


def extract_highres_cutout(input_project, output, name, position, size):
    data_path = os.path.join(input_project, f"S016-{name}", "images", "bdv-n5", "raw.n5")
    data_key = "setup0/timepoint0/s0"

    # positon: from physical position to position in data space
    resolution = [0.04, 0.01, 0.01]  # resolution in micron
    center = [int(pos / res) for pos, res in zip(position[::-1], resolution)]
    print("extract cutout from position")

    # get the cutout in data space:
    # translate the size in micron to a halo and compute the corresponding bounding box
    halo = [int(size / res) // 2 for res in resolution]
    print("cut out block of size", size, "microns, corresponding to", [ha * 2 for ha in halo], "pixel")
    bb = tuple(slice(ce - ha, ce + ha) for ce, ha in zip(center, halo))

    with zarr.open(data_path, mode="r") as f:
        ds = f[data_key]
        data = ds[bb]
    imageio.volwrite(output, data)


def main():
    default_project = os.path.join(
        os.path.split(__file__)[0], "../data"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", "-p", help="The position (json encoded string)", required=True)
    parser.add_argument("-o", "--output", help="where to save the extracted tif cutout", required=True)
    parser.add_argument("-i", "--input", help="path to the input mobie project", default=default_project)
    parser.add_argument("--name", "-n", default="base", help="The dataset name (base or disc)")
    parser.add_argument("-s", "--size", help="size of the cutout in micron", default=15)
    args = parser.parse_args()
    position = json.loads(args.position)
    if isinstance(position, dict):
        position = position["position"]
    assert isinstance(position, list)
    assert len(position) == 3
    extract_highres_cutout(args.input, args.output, args.name, position, args.size)


if __name__ == "__main__":
    main()
