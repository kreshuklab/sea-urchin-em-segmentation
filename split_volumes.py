import os

import mobie
import numpy as np
import pybdv.metadata as bdv_metadata
import vigra

from elf.io import open_file
from pybdv.util import get_scale_factors, absolute_to_relative_scale_factors
from z5py.util import copy_dataset


def get_bounding_boxes():
    mask_path = "./data/S016/images/bdv-n5/S016_aligned_full.n5"
    mask_key = "setup0/timepoint0/s4"

    tmp_mask = "./data/tmp_mask.n5"
    if os.path.exists(tmp_mask):
        with open_file(tmp_mask, "r") as f:
            ds = f["labels"]
            if "bb_starts" in f.attrs:
                starts, stops = f.attrs["bb_starts"], f.attrs["bb_stops"]
                bounding_boxes = [
                    tuple(slice(sta, sto) for sta, sto in zip(start, stop))
                    for start, stop in zip(starts, stops)
                ]
                return ds.shape, bounding_boxes
            ds.n_threads = 8
            labels = ds[:]
    else:
        print("Load mask")
        with open_file(mask_path, "r") as f:
            ds = f[mask_key]
            ds.n_threads = 8
            mask = (ds[:] > 0).astype("uint8")
        print("Label volume")
        labels = vigra.analysis.labelVolumeWithBackground(mask)
        print("Save labels")
        with open_file(tmp_mask, "a") as f:
            f.create_dataset("labels", data=labels, compression="gzip", n_threads=8,
                             chunks=(32, 256, 256))

    label_ids, sizes = np.unique(labels, return_counts=True)
    label_ids, sizes = label_ids[1:], sizes[1:]
    mask_ids = label_ids[np.argsort(sizes)[::-1]][:2]

    starts, stops = [], []
    for mask_id in mask_ids:
        this_mask = np.where(labels == mask_id)
        start = [int(ma.min()) for ma in this_mask]
        stop = [int(ma.max()) + 1 for ma in this_mask]
        starts.append(start)
        stops.append(stop)

    with open_file(tmp_mask, "a") as f:
        f.attrs["bb_starts"] = starts
        f.attrs["bb_stops"] = stops

    bounding_boxes = [
        tuple(slice(sta, sto) for sta, sto in zip(start, stop))
        for start, stop in zip(starts, stops)
    ]
    return labels.shape, bounding_boxes


def extract_sub_volume(in_path, xml_in, out_path, bb, bb_shape, image_name):
    with open_file(in_path, mode="r") as fin:
        in_group = fin["setup0/timepoint0"]
        for name, ds in in_group.items():
            this_shape = ds.shape
            if this_shape != bb_shape:
                scale_factor = [
                    float(ts) / float(bs)
                    for bs, ts in zip(bb_shape, this_shape)
                ]
                this_bb = tuple(
                    slice(int(b.start * sf), int(b.stop * sf)) for b, sf in zip(bb, scale_factor)
                )
            else:
                this_bb = bb

            print("Copy scale", name)
            print("wit bounding box:", this_bb)
            key = f"setup0/timepoint0/{name}"
            copy_dataset(
                in_path, out_path, key, key, n_threads=32, chunks=ds.chunks, roi=this_bb, fit_to_roi=True, verbose=True
            )
    xml_out = out_path.replace(".n5", ".xml")
    unit = bdv_metadata.get_unit(xml_in, setup_id=0)
    resolution = bdv_metadata.get_resolution(xml_in, setup_id=0)

    # write n5 metadata
    scale_factors = get_scale_factors(in_path, setup_id=0)
    scale_factors = absolute_to_relative_scale_factors(scale_factors)
    bdv_metadata.write_n5_metadata(out_path, scale_factors, resolution, setup_id=0, timepoint=0)

    # write bdv metadata
    affine = bdv_metadata.get_affine(xml_in, setup_id=0, timepoint=0)
    attrs = bdv_metadata.get_attributes(xml_in, setup_id=0)
    bdv_metadata.write_xml_metadata(xml_out, out_path, unit, resolution,
                                    is_h5=False, setup_id=0, timepoint=0, setup_name=image_name,
                                    affine=affine, attributes=attrs, overwrite=False, overwrite_data=False,
                                    enforce_consistency=False)
    return xml_out


def create_split_dataset(ds_name, raw_path, mask_path, bb, bb_shape):
    ds_folder = os.path.join("data", ds_name)

    if not os.path.exists(ds_folder):
        mobie.metadata.create_dataset_structure("data", ds_name, ["bdv.n5"])
        mobie.metadata.create_dataset_metadata(ds_folder)
        mobie.metadata.add_dataset("data", ds_name, is_default=ds_name == "S016-base")

    ds_metadata = mobie.metadata.read_dataset_metadata(ds_folder)
    sources = ds_metadata["sources"]
    # copy the raw data
    if "raw" not in sources:
        print("copy raw")
        out_path = os.path.join(ds_folder, "images", "bdv-n5", "raw.n5")
        xml_in = os.path.join(os.path.split(raw_path)[0], "raw.xml")
        xml_path = extract_sub_volume(raw_path, xml_in, out_path, bb, bb_shape, image_name="raw")
        mobie.metadata.add_source_to_dataset(ds_folder, "image", "raw", xml_path)

    # copy the mask
    if "foreground" not in sources:
        print("copy mask")
        out_path = os.path.join(ds_folder, "images", "bdv-n5", "foreground.n5")
        xml_in = os.path.join(os.path.split(mask_path)[0], "foreground.xml")
        xml_path = extract_sub_volume(mask_path, xml_in, out_path, bb, bb_shape, image_name="foreground")
        view = mobie.metadata.get_default_view("image", "foreground", menu_name="mask", contrastLimits=[0, 1])
        mobie.metadata.add_source_to_dataset(ds_folder, "image", "foreground", xml_path, view=view)

    ds_metadata = mobie.metadata.read_dataset_metadata(ds_folder)
    views = ds_metadata["views"]
    if "default" not in views:
        views["default"] = views["raw"]
        ds_metadata["views"] = views
        mobie.metadata.write_dataset_metadata(ds_folder, ds_metadata)


def map_positions(input_ds, output_ds, bb, raw_path):
    input_views = mobie.metadata.read_dataset_metadata(
        os.path.join("data", input_ds)
    )["views"]

    resolution = bdv_metadata.get_resolution(raw_path, setup_id=0)
    print("origin:")
    scale_factor = [4.0, 16.0, 16.0]
    origin = [b.start * sf for b, sf in zip(bb, scale_factor)]
    print("Pixels:", origin)
    origin = [int(orig / res) for orig, res in zip(origin, resolution)]
    print("Phyisical", origin)

    mapped_views = {}
    for name, view in input_views.items():
        if "test-volume" not in name:
            continue
        position = view["viewerTransform"]["position"]
        position = [pos - orig for pos, orig in zip(position, origin)]
        view["viewerTransform"]["position"] = position
        mapped_views[name] = view

    ds_folder = os.path.join("data", output_ds)
    for name, view in mapped_views.items():
        mobie.metadata.add_view_to_dataset(ds_folder, name, view)


def split_volumes():
    raw_path = "./data/S016/images/bdv-n5/S016_aligned_full.n5"
    mask_path = "./data/S016/images/bdv-n5/foreground.n5"
    shape, bounding_boxes = get_bounding_boxes()
    assert len(bounding_boxes) == 2
    # left: base (the interesting part)
    # right: disc
    split_names = ["S016-base", "S016-disc"]
    for ds_name, bb in zip(split_names, bounding_boxes):
        print("Copy dataset", ds_name)
        create_split_dataset(ds_name, raw_path, mask_path, bb, shape)
    map_positions("S016", "S016-base", bounding_boxes[0], "./data/S016-base/images/bdv-n5/raw.xml")


if __name__ == "__main__":
    split_volumes()
