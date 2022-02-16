import argparse
import os
from elf.io import open_file
from torch_em.util import export_bioimageio_model, get_default_citations

from train_nuclei_model import to_binary


def export_enhancer(checkpoint, version):

    halo = [8, 128, 128]
    if "cremi" in checkpoint:
        path = "./data/sampleA.h5"
        name = f"cremi-v{version}"
        key = "volumes/raw"
        description = "Enhancer"
    elif "nuclei" in checkpoint:
        path = "/scratch/pape/platy/nuclei/train_data_nuclei_01.h5"
        name = f"nuclei-v{version}"
        key = "volumes/raw"
        description = "Enhancer"
    elif "cells" in checkpoint:
        path = "./data/cells/train_data_membrane_01.n5"
        name = f"cells-v{version}"
        key = "volumes/raw/s1"
        description = "Enhancer"
    elif "neuropil" in checkpoint:
        path = "./data/neuropil/data.n5"
        name = f"neuropil-v{version}"
        key = "raw"
        description = "Neuropil Prediction"
        halo = [16, 128, 128]

    with open_file(path, "r") as f:
        shape = f[key].shape
        bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(shape, halo))
        input_data = f[key][bb]
    print(bb)
    print(input_data.shape)

    cite = get_default_citations(model="UNet2d")
    tags = ["u-net", "instance-segmentation", "enhancer"]

    output = f"./networks/{name}"
    os.makedirs(output, exist_ok=True)

    doc = "my doc"
    export_bioimageio_model(
        checkpoint, output,
        input_data=input_data,
        name=name,
        description=description,
        authors=[{"name": "Constantin Pape", "affiliation": "EMBL Heidelberg"}],
        tags=tags,
        license="CC-BY-4.0",
        documentation=doc,
        git_repo=None,
        cite=cite,
        input_optional_parameters=False,
        maintainers=[{"github_user": "constantinpape"}],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("-v", "--version", type=int, default=1)
    args = parser.parse_args()
    export_enhancer(args.checkpoint, args.version)


if __name__ == "__main__":
    main()
