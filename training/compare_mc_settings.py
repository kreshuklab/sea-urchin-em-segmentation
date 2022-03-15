import h5py
import napari
import elf.segmentation as eseg


def compute_multicuts(pred, weight_schemes):
    tmp_file = "./pseudo_labels/tmp_prediction.h5"
    with h5py.File(tmp_file, "a") as f:
        print("ws")
        ws, max_id = eseg.stacked_watershed(pred, threshold=0.5, sigma_seeds=2.0, min_size=250)
        print("rag")
        rag = eseg.compute_rag(ws, n_labels=max_id+1)
        print("features")
        feats = eseg.compute_boundary_mean_and_length(rag, pred)
        feats, elen = feats[:, 0], feats[:, -1]
        print("z-mask")
        z_edge_mask = eseg.features.compute_z_edge_mask(rag, ws)
        results = {}
        for name in weight_schemes:
            print("Comute", name, "...")
            if name in f:
                seg = f[name][:]
                results[name] = seg
                continue
            costs = eseg.compute_edge_costs(feats, edge_sizes=elen, z_edge_mask=z_edge_mask, weighting_scheme=name)
            node_labels = eseg.multicut.multicut_kernighan_lin(rag, costs)
            seg = eseg.project_node_labels_to_pixels(rag, node_labels)
            results[name] = seg
            f.create_dataset(name, data=seg, compression="gzip")
    return results


def compare_mc_settings():
    raw_path = "./pseudo_labels/rf2/vanilla/block-3.h5"
    pred_path = "./pseudo_labels/tmp_prediction.h5"

    print("Load prediction ...")
    with h5py.File(pred_path) as f:
        pred = f["roots-segmentor_masked"][:].squeeze().astype("float32")

    weight_schemes = ["all", "none", "xyz", "z"]
    print("Compute multicuts ...")
    segmentations = compute_multicuts(pred, weight_schemes)

    print("Load raw and visualize ...")
    with h5py.File(raw_path, "r") as f:
        raw = f["raw"][:]

    v = napari.Viewer()
    v.add_image(raw)
    for name, seg in segmentations.items():
        v.add_labels(seg, name=name)
    napari.run()


if __name__ == "__main__":
    compare_mc_settings()
