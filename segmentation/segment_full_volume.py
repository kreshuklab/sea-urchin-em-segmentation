import argparse
import json
import os
from copy import deepcopy

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
        ilastik_project=ilp, halo=halo, out_channels=out_channels,
    )
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Ilastik prediction failed"


def predict_boundaries(input_path, input_key,
                       output_path, output_key,
                       mask_path, mask_key, model,
                       tmp_folder, target, max_jobs):
    from cluster_tools.inference import InferenceLocal, InferenceSlurm
    task = InferenceLocal if target == "local" else InferenceSlurm
    config_dir = os.path.join(tmp_folder, "configs")

    global_config = task.default_global_config()
    default_global_config = deepcopy(global_config)

    full_block_shape = (32, 384, 384)
    halo = (8, 64, 64)
    block_shape = tuple(fbs - 2*ha for fbs, ha in zip(full_block_shape, halo))
    print(full_block_shape, block_shape, halo)
    global_config["block_shape"] = block_shape
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "global.config"), "w") as f:
        json.dump(global_config, f)

    conf = task.default_task_config()
    conf["mem_limit"] = 12
    conf["threads_per_job"] = 6
    conf["time_limit"] = 600
    conf["slurm_extras"] = ["#SBATCH --gres=gpu:1"]
    with open(os.path.join(config_dir, "inference.config"), "w") as f:
        json.dump(conf, f)

    output_key_dict = {output_key: [0, 1]}
    t = task(
        tmp_folder=tmp_folder, config_dir=config_dir, max_jobs=max_jobs,
        input_path=input_path, input_key=input_key,
        output_path=output_path, output_key=output_key_dict,
        mask_path=mask_path, mask_key=mask_key,
        checkpoint_path=model, halo=halo, framework="bioimageio"
    )
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Network prediction failed"

    if "partition" in default_global_config:
        default_global_config.pop("partition")
    with open(os.path.join(config_dir, "global.config"), "w") as f:
        json.dump(default_global_config, f)


def blockwise_multicut(path, boundary_key,
                       mask_path, mask_key,
                       tmp_folder, target,
                       max_jobs):
    from cluster_tools import MulticutSegmentationWorkflow
    task = MulticutSegmentationWorkflow

    # TODO set a time limit for the global multicut or use greedy-additive
    config_dir = os.path.join(tmp_folder, "configs")
    # increase runtime and resources for the "map" and "reduce"-like tasks
    map_tasks = ["watershed", "write", "find_uniques", "initial_sub_graphs",
                 "block_edge_features", "probs_to_costs", "merge_edge_features"]
    reduce_tasks = ["find_labeling", "merge_sub_graphs", "map_edge_ids",  # "merge_edge_features",
                    "solve_subproblems", "reduce_problem", "solve_global", "probs_to_costs"]
    configs = task.get_config()
    for task_name in map_tasks:
        conf = configs[task_name]
        conf["time_limit"] = 600 if task_name == "merged_edge_features" else 300
        conf["mem_limit"] = 40 if task_name in ("write", "merge_edge_features") else 4
        conf["threads_per_job"] = 1
        # TODO larger size filter!
        if task_name == "watershed":
            conf.update({"threshold": 0.4})
        with open(os.path.join(config_dir, f"{task_name}.config"), "w") as f:
            json.dump(conf, f)
    for task_name in reduce_tasks:
        conf = configs[task_name]
        conf["time_limit"] = 1200
        conf["mem_limit"] = 256
        conf["threads_per_job"] = 16 if task_name == "solve_subproblems" else 8
        with open(os.path.join(config_dir, f"{task_name}.config"), "w") as f:
            json.dump(conf, f)

    problem_path = os.path.join(tmp_folder, "data.n5")
    t = task(
        tmp_folder=tmp_folder, config_dir=config_dir,
        target=target, max_jobs=max_jobs,
        input_path=path, input_key=boundary_key,
        ws_path=path, ws_key="segmentations/watershed",
        mask_path=mask_path, mask_key=mask_key,
        problem_path=problem_path, node_labels_key="node_labels/multicut",
        output_path=path, output_key="segmentations/multicut",
        max_jobs_multicut=16
    )
    assert luigi.build([t], local_scheduler=True), "Multicut segmentation failed"


def segment_full_volume(
    input_path, mask_path, tmp_path, ilp, model, target, max_jobs, max_jobs_gpu
):
    tmp = os.path.join(tmp_path, "tmp")
    mask_key = "setup0/timepoint0/s0"

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
        rf_path, rf_key, path, boundary_key, mask_path, mask_key, model, tmp, target, max_jobs_gpu
    )
    blockwise_multicut(
        path, boundary_key, mask_path, mask_key, tmp, target, max_jobs
    )


# TODO add postprocessing to apply size filter and maybe more
def main():
    this_folder = os.path.split(__file__)[0]

    # old version with enhancer
    # default_ilp = os.path.join(this_folder, "../ilastik_projects/jil/vol3_3D_pixelclass.ilp")
    # default_model = os.path.join(this_folder, "../networks/cremi-v1.zip")

    # new version with pseudo-label model
    default_ilp = ""
    default_model = os.path.join(this_folder, "../training/networks/roots-segmentor_masked/roots-segmentor_masked.zip")

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="S016-base")
    parser.add_argument("-i", "--ilp", default=default_ilp)
    parser.add_argument("-m", "--model", default=default_model)
    parser.add_argument("-t", "--target", default="slurm")
    parser.add_argument("--max_jobs", default=200, type=int)
    parser.add_argument("--max_jobs_gpu", default=12, type=int)

    args = parser.parse_args()
    dataset = args.dataset

    input_path = os.path.abspath(os.path.join(this_folder, f"../data/{dataset}/images/bdv-n5/raw.n5"))
    assert os.path.exists(input_path), input_path
    mask_path = os.path.abspath(os.path.join(this_folder, f"../data/{dataset}/images/bdv-n5/foreground.n5"))
    assert os.path.exists(mask_path), mask_path

    tmp_path = f"/scratch/pape/jils_project/seg_{dataset}"

    os.makedirs(tmp_path, exist_ok=True)
    segment_full_volume(input_path, mask_path, tmp_path, args.ilp, args.model,
                        args.target, args.max_jobs, args.max_jobs_gpu)


if __name__ == "__main__":
    main()
