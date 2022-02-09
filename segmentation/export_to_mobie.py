import os
import mobie
from pybdv.metadata import get_resolution
from pybdv.util import get_scale_factors, absolute_to_relative_scale_factors


# export the segmentation to mobie / over-write if segmentation is already there
def export_to_mobie():
    input_path = "/scratch/pape/jils_project/full_seg/data.n5"
    input_key = "segmentations/multicut"

    root = "../data"
    ds_name = "S016"
    ds_folder = os.path.join(root, ds_name)
    seg_name = "neurons"
    metadata = mobie.metadata.read_dataset_metadata(ds_folder)
    # not implemented yet
    if seg_name in metadata.sources:
        raise RuntimeError

    raw_path = "../data/S016/images/bdv-n5/S016_aligned_full.n5"
    xml_path = "../data/S016/images/bdv-n5/raw.xml"
    resolution = get_resolution(xml_path, setup_id=0)
    scale_factors = absolute_to_relative_scale_factors(
        get_scale_factors(raw_path, setup_id=0)
    )
    chunks = [96] * 3
    target = "slurm"
    max_jobs = 400
    mobie.add_segmentation(input_path, input_key, root, ds_name, seg_name,
                           resolution=resolution, scale_factors=scale_factors,
                           chunks=chunks, menu_name="segmentation", add_default_table=False,
                           target=target, max_jobs=max_jobs)


if __name__ == "__main__":
    export_to_mobie()
