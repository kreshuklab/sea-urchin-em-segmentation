import argparse
import os
import h5py
from torch_em.util import export_bioimageio_model, get_default_citations


def export_pseudolabel_model(checkpoint):
    path = "./pseudo_labels/rf/vanilla/block-0.h5"
    with h5py.File(path, "r") as f:
        input_data = f["raw"][:16, :256, :256]

    name = "_".join(os.path.basename(checkpoint).split("_")[1:])
    description = "Neuron boundary segmentation"
    tags = ["u-net", "neuron-segmentation", "segmentation"]

    # eventually we should refactor the citation logic
    cite = get_default_citations(model="UNet3d")

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
    args = parser.parse_args()
    export_pseudolabel_model(args.checkpoint)


if __name__ == "__main__":
    main()
