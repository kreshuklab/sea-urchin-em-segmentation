import napari
from elf.io import open_file
from torch_em.util import get_trainer
from torch_em.util.prediction import predict_with_halo


def predict_checkpoint(raw, ckpt, name):
    block_shape = [32, 256, 256]
    halo = [4, 16, 16]
    tmp_path = "./pseudo_labels/tmp_prediction.h5"
    with open_file(tmp_path, "a") as f:
        if name in f:
            return f[name][:]
        model = get_trainer(ckpt).model
        pred = predict_with_halo(raw, model, [4, 5, 6, 7], block_shape, halo)
        f.create_dataset(name, data=pred, compression="gzip")
    return pred


def check_all_predictions():
    checkpoints = [
        "checkpoints/precomputed_roots-rf_masked",
        "checkpoints/precomputed_roots-autocontext_masked",
        "checkpoints/precomputed_roots-autocontext-rf_masked",
    ]
    block_id = 2
    data_path = f"./pseudo_labels/rf/vanilla/block-{block_id}.h5"
    with open_file(data_path, "r") as f:
        raw = f["raw"][:]

    predictions = {}
    for ckpt in checkpoints:
        name = "_".join(ckpt.split("_")[1:])
        predictions[name] = predict_checkpoint(raw, ckpt, name)

    v = napari.Viewer()
    v.add_image(raw)
    for name, pred in predictions.items():
        v.add_image(pred, name=name)
    napari.run()


def compare_predictions():
    ckpt = "checkpoints/precomputed_roots-rf_masked"
    block_id = 2
    data_path = f"./pseudo_labels/rf/vanilla/block-{block_id}.h5"
    with open_file(data_path, "r") as f:
        raw = f["raw"][:]
        enhancer = f["pseudo-labels"][:]

    name = "_".join(ckpt.split("_")[1:])
    retrained = predict_checkpoint(raw, ckpt, name)

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(enhancer)
    v.add_image(retrained)
    napari.run()


if __name__ == "__main__":
    compare_predictions()
    # check_all_predictions()
