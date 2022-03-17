import argparse
import os

import z5py
from paintera_tools import convert_to_paintera_format


def to_paintera(out_path, target, max_jobs, use_label_multiset):
    tmp_folder = "./tmp_paintera/tmp"
    os.makedirs(tmp_folder, exist_ok=True)

    # create the output file
    z5py.File(out_path, "a")

    # link the raw data and initial labels
    raw_path = "/g/emcf/pape/jil/data/S016-base/images/bdv-n5/raw.n5/setup0/timepoint0"
    raw_link = os.path.join(out_path, "raw")
    if not os.path.exists(raw_link):
        os.symlink(raw_path, raw_link)

    label_path = "/g/emcf/pape/jil/data/S016-base/images/bdv-n5/neurons.n5/setup0/timepoint0/s0"
    label_link = os.path.join(out_path, "labels")
    if not os.path.exists(label_link):
        os.symlink(label_path, label_link)

    restrict_sets = [-1, -1, 5, 3, 2] if use_label_multiset else None
    resolution = [0.04, 0.01, 0.01]
    convert_to_paintera_format(
        out_path, "raw", "labels", "paintera-labels",
        label_scale=0, resolution=resolution,
        tmp_folder=tmp_folder, target=target, max_jobs=max_jobs,
        max_threads=12, convert_to_label_multisets=use_label_multiset,
        restrict_sets=restrict_sets, label_block_mapping_compression="gzip"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_path", required=True)
    parser.add_argument("-t", "--target", default="local")
    parser.add_argument("-m", "--max_jobs", default=24)
    parser.add_argument("-u", "--use_label_multiset", default=0)
    args = parser.parse_args()
    to_paintera(args.out_path, args.target, args.max_jobs, use_label_multiset=bool(args.use_label_multiset))


if __name__ == "__main__":
    main()
