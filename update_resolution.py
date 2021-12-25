import z5py
from pybdv.metadata import (write_size_and_resolution, get_size,
                            get_affine, write_affine)


def update_n5():
    path = "./data/S016/images/bdv-n5/S016_aligned_full.n5"
    with z5py.File(path, "a") as f:
        g = f["setup0/timepoint0"]
        new_attrs = {
            "pixelResolution": {"dimensions": [0.01, 0.01, 0.04], "unit": "micrometer"},
            "resolution": [0.01, 0.01, 0.04],
            "units": ["micrometer", "micrometer", "micrometer"]
        }
        g.attrs.update(new_attrs)
        for k, v in g.attrs.items():
            print(k, v)


def update_xml():
    path = "./data/S016/images/bdv-n5/raw.xml"
    size = get_size(path, 0)
    resolution = [0.04, 0.01, 0.01]
    write_size_and_resolution(path, 0, size, resolution)
    aff = get_affine(path, 0)["affine0"]
    aff = [val / 1000 for val in aff]
    print(aff)
    write_affine(path, 0, aff, overwrite=True)


if __name__ == "__main__":
    update_n5()
    print()
    print()
    print()
    update_xml()
