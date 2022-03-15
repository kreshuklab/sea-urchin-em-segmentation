import argparse
import os
import mobie
from pybdv.metadata import get_resolution, get_data_path
from pybdv.util import get_scale_factors, absolute_to_relative_scale_factors
from shutil import move


def update_segmentation(input_path, input_key, output_path,
                        resolution, scale_factors, chunks,
                        source_name, target, max_jobs):
    name = os.path.basename(output_path)
    print("Updating segmentation:", name)
    bkp_path = f"{name}.bkp"
    if os.path.exists(output_path):
        print("Moving", output_path, "to", bkp_path)
        move(output_path, bkp_path)
    assert not os.path.exists(output_path)

    tmp_folder = f"./tmp_update_{os.path.splitext(name)[0]}"
    mobie.import_data.import_segmentation(
        input_path, input_key, output_path,
        resolution, scale_factors, chunks,
        source_name=source_name,
        tmp_folder=tmp_folder, target=target, max_jobs=max_jobs)

    print("The segmentation was successfully updated and the previous version moved to", bkp_path)


# export the segmentation to mobie / over-write if segmentation is already there
def export_to_mobie(dataset, target, max_jobs):
    input_path = f"/scratch/pape/jils_project/seg_{dataset}/data.n5"
    input_key = "segmentations/multicut"

    root = "../data"
    ds_folder = os.path.join(root, dataset)
    seg_name = "neurons"
    metadata = mobie.metadata.read_dataset_metadata(ds_folder)

    raw_path = f"../data/{dataset}/images/bdv-n5/raw.n5"
    xml_path = f"../data/{dataset}/images/bdv-n5/raw.xml"
    resolution = get_resolution(xml_path, setup_id=0)
    scale_factors = absolute_to_relative_scale_factors(
        get_scale_factors(raw_path, setup_id=0)
    )
    chunks = [96] * 3

    if seg_name in metadata["sources"]:
        source = metadata["sources"][seg_name]["segmentation"]
        output_path = os.path.join(root, ds_folder, source["imageData"]["bdv.n5"]["relativePath"])
        assert os.path.exists(output_path), output_path
        output_path = get_data_path(output_path, return_absolute_path=True)
        update_segmentation(
            input_path, input_key, output_path, resolution, scale_factors, chunks, seg_name, target, max_jobs
        )
    else:
        mobie.add_segmentation(input_path, input_key, root, dataset, seg_name,
                               resolution=resolution, scale_factors=scale_factors,
                               chunks=chunks, menu_name="segmentation", add_default_table=False,
                               target=target, max_jobs=max_jobs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="S016-base")
    parser.add_argument("-t", "--target", default="slurm")
    parser.add_argument("-m", "--max_jobs", default=100, type=int)
    args = parser.parse_args()
    export_to_mobie(args.dataset, args.target, args.max_jobs)


if __name__ == "__main__":
    main()
