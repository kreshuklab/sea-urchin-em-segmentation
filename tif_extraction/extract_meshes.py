import os

import numpy as np
import zarr
import elf.mesh.io as mesh_io
from elf.mesh import marching_cubes


# TODO make proper script
def main():
    scale = 4

    # S0: 0.04, 0.01, 0.01
    # S1: 0.04, 0.02, 0.02
    # S2: 0.04, 0.04, 0.04
    # S3: 0.08, 0.08, 0.08
    # S4: 0.16, 0.16, 0.16
    resolution = [0.16, 0.16, 0.16]
    position = {"position": [98.21658841446563, 68.24941392726335, 35.55837586643707], "timepoint": 0}

    name = "base"
    data_path = os.path.join("..", "data", f"S016-{name}", "images", "bdv-n5", "neurons.n5")
    data_key = f"setup0/timepoint0/s{scale}"

    position = position["position"]
    center = [int(pos / res) for pos, res in zip(position[::-1], resolution)]
    # TODO the halo needs to be dynamically adjusted based on the actual segment extents
    halo = [20, 100, 100]
    bb = tuple(slice(ce - ha, ce + ha) for ce, ha in zip(center, halo))

    with zarr.open(data_path, mode="r") as f:
        ds = f[data_key]
        data = ds[bb]

    center = tuple(sh // 2 for sh in data.shape)
    seg_id = data[center]
    print("Seg-id:", seg_id)
    mask = data == seg_id

    # import napari
    # v = napari.Viewer()
    # v.add_labels(data)
    # v.add_labels(mask)
    # napari.run()

    verts, faces, normals = marching_cubes(mask)
    offset = np.array([b.start for b in bb])
    verts += offset

    mesh_io.write_obj("mesh.obj", verts, faces, normals)
    mesh_io.write_ply("mesh.ply", verts, faces)


if __name__ == "__main__":
    main()
