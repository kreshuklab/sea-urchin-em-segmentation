import argparse
import json
import os
import mobie
from pybdv.metadata import get_resolution
from pybdv.util import get_scale_factors, absolute_to_relative_scale_factors


# export the segmentation to mobie / over-write if segmentation is already there
def export_to_mobie(target, max_jobs):
    input_path = "/scratch/pape/jils_project/full_seg/data.n5"
    input_key = "segmentations/multicut"

    root = "../data"
    ds_name = "S016"
    ds_folder = os.path.join(root, ds_name)
    seg_name = "neurons"
    metadata = mobie.metadata.read_dataset_metadata(ds_folder)
    # not implemented yet
    if seg_name in metadata["sources"]:
        raise RuntimeError

    # FIXME this is overwritten later
    conf_path = "./tmp_S016_neurons/configs/downscaling.config"
    with open(conf_path, "r") as f:
        conf = json.load(f)
    conf["time_limit"] = 300
    with open(conf_path, "w") as f:
        json.dump(conf, f)

    raw_path = "../data/S016/images/bdv-n5/S016_aligned_full.n5"
    xml_path = "../data/S016/images/bdv-n5/raw.xml"
    resolution = get_resolution(xml_path, setup_id=0)
    scale_factors = absolute_to_relative_scale_factors(
        get_scale_factors(raw_path, setup_id=0)
    )
    print()
    print(resolution)
    print(scale_factors)
    print()
    chunks = [96] * 3
    mobie.add_segmentation(input_path, input_key, root, ds_name, seg_name,
                           resolution=resolution, scale_factors=scale_factors,
                           chunks=chunks, menu_name="segmentation", add_default_table=False,
                           target=target, max_jobs=max_jobs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default="slurm")
    parser.add_argument("-m", "--max_jobs", default=200, type=int)
    args = parser.parse_args()
    export_to_mobie(args.target, args.max_jobs)


if __name__ == "__main__":
    main()
