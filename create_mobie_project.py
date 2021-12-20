import os
from shutil import rmtree

import h5py
import mobie
import numpy as np
from mobie.xml_utils import copy_xml_with_newpath


def create_project():
    path = "./data.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.random.randint(0, 255, size=(128, 128, 128)).astype("uint8"))

    resolution = [1., 1., 1.]
    scale_factors = [[2, 2, 2]]
    mobie.add_image(path, "data", "./data", "S016", "raw", resolution,
                    scale_factors=scale_factors, chunks=(64, 64, 64), menu_name="em")
    rmtree("./data/S016/images/bdv-n5/raw.n5")


def main():
    # create project with fake data
    if not os.path.exists("./data"):
        create_project()

    # modify the project with the correct xml path
    xml_src = "/g/emcf/ronchi/Lueter-Esther/S016_21-12-08_10nm_final/aligned/20211215/S016_aligned_full.xml"
    xml_trg = "./data/S016/images/bdv-n5/raw.xml"
    data_path = "/g/emcf/ronchi/Lueter-Esther/S016_21-12-08_10nm_final/aligned/20211215/S016_aligned_full.n5"

    rel_path = os.path.relpath(data_path, "./data/S016/images/bdv-n5")
    copy_xml_with_newpath(xml_src, xml_trg, rel_path)


if __name__ == "__main__":
    main()
