import argparse
import os
import h5py
from torch_em.util import export_bioimageio_model, get_default_citations


def export_enhancer(version):
    path = "./data/sampleA.h5"
    with h5py.File(path, "r") as f:
        input_data = f["volumes/raw"][:16, :256, :256]

    checkpoint = f"./checkpoints/cremi3d-v{version}"

    name = f"cremi-v{version}"
    description = "Affinity segmentation"
    tags = ["u-net", "cell-segmentation", "segmentation", "phase-contrast", "livecell"]

    # eventually we should refactor the citation logic
    cite = get_default_citations(model="UNet2d")

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


# TODO
def export_pseudolabel(version):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", required=True, type=int)
    parser.add_argument("-p", "--pseudolabel", type=int, default=0)
    args = parser.parse_args()
    if args.pseudolabel:
        export_pseudolabel(args.version)
    else:
        export_enhancer(args.version)


if __name__ == "__main__":
    main()
