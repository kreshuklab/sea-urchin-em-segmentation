# autocontext prediction does not work with ilastik API yet :(
# so need to fall back on headless mode
import argparse
import os
import subprocess
import h5py
import numpy as np


def _predict_impl(path, out_path, ilp):
    # look for run_ilastik.sh or ilastik.py
    ilastik_folder = "/g/kreshuk/pape/Work/software/src/ilastik/ilastik-1.4.0b8-Linux"
    ilastik_exe = os.path.join(ilastik_folder, "run_ilastik.sh")
    assert os.path.exists(ilastik_exe), ilastik_exe
    input_str = f"{path}/raw"
    cmd = [ilastik_exe, "--headless",
           "--project=%s" % ilp,
           "--output_format=compressed hdf5",
           "--raw_data=%s" % input_str,
           "--output_filename_format=%s" % out_path,
           "--readonly=1"]
    print("Run ilastik prediction ...")
    subprocess.run(cmd)


def predict_volume(path, output_path, ilp, subtract_bg=True):
    prefix = os.path.splitext(os.path.basename(ilp))[0]
    key = f"{prefix}/rf_pred"
    if os.path.exists(path):
        with h5py.File(path, "r") as f:
            if key in f:
                return
    tmp_path = "./ilastik_tmp.h5"
    _predict_impl(path, tmp_path, ilp)
    with h5py.File(tmp_path, "r") as f:
        data = f["exported_data"][:]
    bg, pred = data[..., 0], data[..., 1]
    if subtract_bg:
        pred = np.clip(pred - bg, 0, 1)
    with h5py.File(output_path, "a") as f:
        f.create_dataset(key, data=pred, compression="gzip")
    os.remove(tmp_path)


def main():
    this_folder = os.path.split(__file__)[0]
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vol_id", type=int, required=True)
    parser.add_argument("-i", "--ilp", required=True)
    args = parser.parse_args()
    path = os.path.join(this_folder, f"../test_volumes/vol{args.vol_id}.h5")
    os.makedirs("data", exist_ok=True)
    out_path = f"./data/vol{args.vol_id}.h5"
    predict_volume(path, out_path, args.ilp)


if __name__ == "__main__":
    main()
