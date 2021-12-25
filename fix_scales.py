import z5py


def fix_scales():
    with z5py.File("./data/S016/images/bdv-n5/S016_aligned_full.n5") as f:
        scales = f["setup0/timepoint0"].attrs["scales"]
        f["setup0"].attrs["downsamplingFactors"] = scales
        print(list(f["setup0"].attrs["downsamplingFactors"]))

        g = f["setup0/timepoint0"]
        n_scales = len(g)
        assert n_scales == len(scales)
        for ii in range(n_scales):
            g[f"s{ii}"].attrs["downsamplingFactors"] = scales[ii]


if __name__ == "__main__":
    fix_scales()
