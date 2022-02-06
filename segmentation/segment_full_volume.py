import argparse
import json
import os

import luigi


def predict_ilastik(input_path, input_key,
                    output_path, output_key,
                    mask_path, mask_key, ilp,
                    tmp_folder, target, max_jobs):
    from cluster_tools.ilastik import IlastikPredictionWorkflow
    config_dir = os.path.join(tmp_folder, "configs")
    os.makedirs(config_dir, exist_ok=True)

    conf = IlastikPredictionWorkflow.get_config()["prediction"]
    conf["mem_limit"] = 12
    conf["time_limit"] = 600
    with open(os.path.join(config_dir, "prediction.config"), "w") as f:
        json.dump(conf, f)

    halo = [2, 32, 32]
    out_channels = [1]

    t = IlastikPredictionWorkflow(
        tmp_folder=tmp_folder, config_dir=config_dir,
        target=target, max_jobs=max_jobs,
        input_path=input_path, input_key=input_key,
        output_path=output_path, output_key=output_key,
        mask_path=mask_path, mask_key=mask_key,
        ilastik_project=ilp, halo=halo, out_channels=out_channels
    )
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Ilastik prediction failed"


def predict_boundaries(input_path, input_key,
                       output_path, output_key,
                       mask_path, mask_key, model,
                       tmp_folder, target, max_jobs):
    import bioimageio.core
    from cluster_tools.inference import InferenceLocal, InferenceSlurm
    task = InferenceLocal if target == "local" else InferenceSlurm
    config_dir = os.path.join(tmp_folder, "configs")

    model_spec = bioimageio.core.load_resource_description(model)
    halo = model_spec.outputs[0].halo

    # TODO expose convenience function in bioimageio for this
    # make sure the block shape fits the network
    # min_shape = model_spec.inputs[0].shape.min[2:]
    # step = model_spec.inputs[0].shape.step[2:]
    # block_shape =
    # full_block_shape = [bs + 2 * ha]

    conf = task.default_task_config()
    conf["mem_limit"] = 8
    conf["time_limit"] = 600
    with open(os.path.join(config_dir, "inference.config"), "w") as f:
        json.dump(conf, f)

    t = task(
        tmp_folder=tmp_folder, config_dir=config_dir, max_jobs=max_jobs,
        input_path=input_path, input_key=input_key,
        output_path=output_path, output_key=output_key,
        mask_path=mask_path, mask_key=mask_key,
        checkpoint_path=model, halo=halo, framework="bioimageio"
    )
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Network prediction failed"


def blockwise_multicut(path, boundary_key,
                       mask_path, mask_key,
                       tmp_folder, target, max_jobs):
    from cluster_tools import MulticutSegmentationWorkflow
    config_dir = os.path.join(tmp_folder, "configs")
    problem_path = os.path.join(tmp_folder, "data.n5")
    task = MulticutSegmentationWorkflow(
        tmp_folder=tmp_folder, config_dir=config_dir,
        target=target, max_jobs=max_jobs,
        input_path=path, input_key=boundary_key,
        ws_path=path, ws_key="segmentations/watershed",
        problem_path=problem_path, node_labels_key="node_labels/multicut",
        output_path=path, output_key="segmentations/multicut"
    )
    assert luigi.build([task], local_scheduler=True), "Multicut segmentation failed"


def segment_full_volume(input_path, mask_path, tmp_path, ilp, model, use_bb, target, max_jobs):
    tmp = os.path.join(tmp_path, "tmp")
    mask_key = "setup0/timepoint0/s0"

    # TODO find proper bounding box
    if use_bb:
        pass

    path = os.path.join(tmp_path, "data.n5")
    if ilp == "":  # network prediction directly from raw
        rf_path = input_path
        rf_key = "setup0/timepoint0/s0"
    else:
        key = "setup0/timepoint0/s0"
        rf_key = "predictions/rf"
        predict_ilastik(
            input_path, key, path, rf_key, mask_path, mask_key, ilp, tmp, target, max_jobs
        )
        rf_path = path

    boundary_key = "predictions/enhancer"
    predict_boundaries(
        rf_path, rf_key, path, boundary_key, mask_path, mask_key, model, tmp, target, max_jobs
    )
    blockwise_multicut(
        path, boundary_key, mask_path, mask_key, tmp, target, max_jobs
    )


def main():
    this_folder = os.path.split(__file__)[0]
    default_ilp = os.path.join(this_folder, "../ilastik_projects/jil/vol3_3D_pixelclass.ilp")
    default_model = os.path.join(this_folder, "../networks/cremi-v1.zip")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ilp", default=default_ilp)
    parser.add_argument("-m", "--model", default=default_model)
    parser.add_argument("-b", "--use_bb", default=0, type=int)
    parser.add_argument("-t", "--target", default="slurm")
    parser.add_argument("--max_jobs", default=200, type=int)

    args = parser.parse_args()
    use_bb = bool(args.use_bb)

    input_path = os.path.abspath(os.path.join(this_folder, "../data/S016/images/bdv-n5/S016_aligned_full.n5"))
    assert os.path.exists(input_path), input_path
    mask_path = os.path.abspath(os.path.join(this_folder, "../data/S016/images/bdv-n5/foreground.n5"))
    assert os.path.exists(mask_path), mask_path

    tmp_path = "/scratch/pape/jils_project/full_seg"
    if use_bb:
        tmp_path += "_with_bb"

    os.makedirs(tmp_path, exist_ok=True)
    segment_full_volume(input_path, mask_path, tmp_path, args.ilp, args.model, use_bb,
                        args.target, args.max_jobs)


if __name__ == "__main__":
    main()
