import argparse
import json
import os

import numpy as np
import zarr

import elf.mesh.io as mesh_io
from elf.mesh import marching_cubes


def check_assignments(paintera_path, project_path):
    # assignment_key = "paintera-labels/fragment-segment-assignment"
    # with open_file(paintera_path, "r") as f:
    #     assignments = f[assignment_key][:]
    paintera_attributes = os.path.join(project_path, "attributes.json")
    with open(paintera_attributes, "r") as f:
        attrs = json.load(f)
    sources = attrs["paintera"]["sourceInfo"]["sources"]
    assignments = None
    for source in sources:
        if source["type"] != "org.janelia.saalfeldlab.paintera.state.label.ConnectomicsLabelState":
            continue
        assignments = source["state"]["backend"]["data"]["fragmentSegmentAssignment"]["actions"]

    if assignments is None:
        raise RuntimeError("parsing assignments failed")
    if len(assignments) > 0:
        raise RuntimeError(f"Can't handle non empty assignments, got {len(assignments)} assignments")


# TODO only load paintera tools stuff if needed
def export_paintera_segmentation(paintera_path, project_path, output_path, scale, tmp_folder):
    # as usual I don't understand what exactly paintera does...
    # check_assignments(paintera_path, project_path)
    from paintera_tools.serialize import serialize_from_commit
    serialize_from_commit(paintera_path, "paintera-labels", output_path, f"s{scale}",
                          tmp_folder=tmp_folder, max_jobs=8, target="local",
                          scale=scale)


# resolutions in micron (zyx)
# S0: 0.04, 0.01, 0.01
# S1: 0.04, 0.02, 0.02
# S2: 0.04, 0.04, 0.04
# S3: 0.08, 0.08, 0.08
# S4: 0.16, 0.16, 0.16
def get_resolution(scale):
    resolutions = {
        0: [40, 10, 10],
        1: [40, 20, 20],
        2: [40, 40, 40],
        3: [80, 80, 80],
        4: [160, 160, 160],
    }
    return resolutions[scale]


def extract_mesh(seg_path, seg_key, seg_ids, scale, project, to_xyz=True):
    print("Loading segmentation")
    with zarr.open(seg_path, mode="r") as f:
        ds = f[seg_key]
        ds.n_threads = 8
        seg = ds[:]

    resolution = get_resolution(scale)
    for seg_id in seg_ids:
        print("computing mesh for segment_id", seg_id)
        mask = seg == seg_id
        nfg = mask.sum()
        print("Foreground:", nfg, "/", mask.size, "(", float(nfg) / mask.size, ")")
        verts, faces, normals = marching_cubes(mask)
        verts *= np.array(resolution)
        if to_xyz:
            verts = verts[:, ::-1]
        print(verts.min(axis=0))
        print(verts.max(axis=0))
        # offset = np.array([b.start for b in bb])
        # verts += offset
        print("saving mesh for segment_id", seg_id)
        mesh_io.write_ply(f"segment-{seg_id}-{project}.ply", verts, faces)


def require_segment_ids(paintera_path, ids):
    with zarr.open(paintera_path, "r") as f:
        assignments = f["paintera-labels/fragment-segment-assignment"][:].T
    assignment_dict = dict(zip(assignments[:, 0], assignments[:, 1]))
    segment_ids = [assignment_dict.get(idd, idd) for idd in ids]
    return segment_ids


def export_paintera(output_path, scale, project):
    assert project in ("new", "old")
    if project == "new":
        paintera_path = "/g/emcf/pape/jil/segmentation/paintera/neurons.n5"
        project_path = "/g/emcf/carl/painteraprojekte/projektnewneurons"
        tmp_folder = "./tmp_export_new"
    else:
        paintera_path = "/g/emcf/pape/jil/segmentation/paintera/old.n5"
        project_path = "/g/emcf/carl/painteraprojekte/projektoldneurons"
        tmp_folder = "./tmp_export_old"
    export_paintera_segmentation(paintera_path, project_path, output_path, scale, tmp_folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--segment_ids", nargs="+", type=int, required=True)
    parser.add_argument("-p", "--project", default="new")
    parser.add_argument("-s", "--scale", default=4, type=int)

    args = parser.parse_args()
    segment_ids = args.segment_ids
    project = args.project
    scale = args.scale

    path = "./paintera-export-{project}.n5"
    if not os.path.exists(path):
        export_paintera(path, scale, project)

    extract_mesh(path, f"s{scale}", segment_ids, scale, project)


if __name__ == "__main__":
    main()
