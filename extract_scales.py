import os
import h5py
import luigi
import z5py


def make_scales():
    from cluster_tools.downscaling import DownscalingWorkflow
    os.makedirs("./downscaled_data", exist_ok=True)
    out_path = "./downscaled_data/data.n5"
    scale_factors = [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    task = DownscalingWorkflow(
        input_path="./data/S016/images/bdv-n5/S016_aligned_full.n5", input_key="setup0/timepoint0/s0",
        scale_factors=scale_factors, halos=scale_factors,
        output_path=out_path, metadata_format="bdv.n5",
        tmp_folder="./tmp_downscale", config_dir="./tmp_downscale/configs",
        max_jobs=24, target="local"
    )
    luigi.build([task], local_scheduler=True)


def extract_scale(scale):
    path = "./data/S016/images/bdv-n5/S016_aligned_full.n5"
    print("Load data for scale", scale)
    with z5py.File(path, "r") as f:
        key = f"setup0/timepoint0/s{scale}"
        ds = f[key]
        ds.n_threads = 8
        data = ds[:]
    print("Save data for scale", scale)
    out_path = f"./downscaled_data/volume_s{scale}.h5"
    with h5py.File(out_path, "a") as f:
        f.create_dataset("raw", data=data, compression="gzip")


def check_scale(scale):
    path = "./downscaled_data/data.n5"
    with z5py.File(path, "r") as f:
        key = f"setup0/timepoint0/s{scale}"
        ds = f[key]
        shape = ds.shape
        print("Scale:", scale)
        print("Shape:", shape)


def check_scales():
    scales = [3, 4, 5]
    for scale in scales:
        check_scale(scale)


def extract_scales():
    scales = [3, 4, 5]
    for scale in scales:
        extract_scale(scale)


if __name__ == "__main__":
    # make_scales()
    # check_scales()
    extract_scales()
