import argparse
import os

import imageio
import zarr


def extract_lowres_volume(input_project, output, name, scale):
    data_path = os.path.join(input_project, f"S016-{name}", "images", "bdv-n5", "raw.n5")
    data_key = f"setup0/timepoint0/s{scale}"
    with zarr.open(data_path, mode="r") as f:
        ds = f[data_key]
        data = ds[:]
    imageio.volwrite(output, data)


def main():
    default_project = os.path.join(
        os.path.split(__file__)[0], "../data"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="where to save the extracted tif volume")
    parser.add_argument("-i", "--input", help="path to the input mobie project", default=default_project)
    parser.add_argument("--name", "-n", default="base", help="The dataset name (base or disc)")
    parser.add_argument("--scale", "-s", type=int, default=4,
                        help="The scale level at which to export the full volume")
    args = parser.parse_args()

    if args.output is None:
        output = f"volume_{args.name}_s{args.scale}.tif"
    else:
        output = args.output
    extract_lowres_volume(args.input, output, args.name, args.scale)


if __name__ == "__main__":
    main()
