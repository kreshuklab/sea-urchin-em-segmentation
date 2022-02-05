import argparse
import os

import luigi


def predict_ilastik(input_path, input_key,
                    output_path, output_key,
                    mask_path, mask_key,
                    ilp, tmp_folder):
    pass


def predict_boundaries(input_path, input_key,
                       output_path, output_key,
                       mask_path, mask_key,
                       model, tmp_folder):
    pass


def blockwise_multicut(path, boundary_key,
                       mask_path, mask_key,
                       tmp_folder, target, max_jobs):
    from cluster_tools import MulticutSegmentationWorkflow
    config_dir = os.path.join(tmp_folder, "config")
    problem_path = os.path.join(tmp_folder, "data.n5")
    task = MulticutSegmentationWorkflow(
        tmp_folder=tmp_folder, config_dir=config_dir,
        input_path=path, input_key=boundary_key,
        ws_path=path, ws_key="segmentations/watershed",
        problem_path=problem_path, node_labels_key="node_labels/multicut",
        output_path=path, output_key="segmentations/multicut"
    )
    assert luigi.build([task], local_scheduler=True), "Multicut segmentation failed"


def segment_full_volume(input_path, tmp_path, ilp, model, use_bb, use_mask):
    tmp = os.path.join(tmp_path, "tmp")

    # TODO find proper bounding box
    if use_bb:
        pass
    # TODO create masks
    if use_mask:
        mask_path = ""
        mask_key = ""
    else:
        mask_path, mask_key = "", ""

    path = os.path.join(tmp_path, "data.n5")
    if ilp == "":  # network prediction directly from raw
        rf_path = input_path
        rf_key = "setup0/timepoint0/s0"
    else:
        key = "setup0/timepoint0/s0"
        rf_key = "predictions/rf"
        predict_ilastik(input_path, key, path, rf_key, mask_path, mask_key, ilp, tmp)
        rf_path = path

    boundary_key = "predictions/enhancer"
    predict_boundaries(rf_path, rf_key, path, boundary_key, mask_path, mask_key, model, tmp)

    blockwise_multicut(path, boundary_key, mask_path, mask_key, tmp)


def main():
    this_folder = os.path.split(__file__)[0]
    default_ilp = os.path.join(this_folder, "../ilastik_projects/boundary_prediction_v2.ilp")
    default_model = os.path.join(this_folder, "../networks/cremi-v1.zip")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ilp", default=default_ilp)
    parser.add_argument("-m", "--model", default=default_model)
    parser.add_argument("-b", "--use_bb", default=0, type=int)
    parser.add_argument("-u", "--use_mask", default=0, type=int)

    args = parser.parse_args()
    use_bb = bool(args.use_bb)
    use_mask = bool(args.use_mask)

    input_path = os.path.abspath(this_folder, "../data/S016/images/bdv-n5/raw.n5")
    tmp_path = "/scratch/pape/jils_project/full_seg"
    if use_bb:
        tmp_path += "_with_bb"
    if use_mask:
        tmp_path += "_with_mask"

    os.makedirs(tmp_path, exist_ok=True)
    segment_full_volume(input_path, tmp_path, args.ilp, args.model, use_bb, use_mask)


if __name__ == "__main__":
    main()
