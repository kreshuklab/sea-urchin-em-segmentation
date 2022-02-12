import os
from glob import glob

import numpy as np
import torch_em
from torch_em.model import AnisotropicUNet


def get_loader(split, patch_shape, args):
    raw_key = "raw"
    label_key = "pseudo-labels"

    paths, rois = [], []
    for root_name in args.roots:
        assert root_name in ("autocontext", "rf")
        root = f"./pseudo_labels/{root_name}"
        if args.masked:
            root = os.path.join(root, "masked")
        else:
            root = os.path.join(root, "vanilla")
        files = glob(os.path.join(root, "*.h5"))
        files.sort()
        assert len(files) > 0
        if split == "train":
            paths.extend(files)
            n_files = len(files)
            rois.extend((n_files - 1) * [np.s_[:, :, :]])
            rois.append(np.s_[0:75, :, :])
        else:
            paths.append(files[-1])
            rois.append(np.s_[75:, :, :])
    assert len(paths) == len(rois)

    n_samples = 500 if split == "train" else 25

    loader = torch_em.default_segmentation_loader(
        paths, raw_key, paths, label_key, batch_size=1, patch_shape=patch_shape,
        ndim=3, n_samples=n_samples
    )
    return loader


def train_precomuted(args):
    model = AnisotropicUNet(in_channels=1, out_channels=1, final_activation="Sigmoid",
                            scale_factors=[[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]])

    patch_shape = (32, 384, 384)
    train_loader = get_loader("train", patch_shape, args)
    val_loader = get_loader("val", patch_shape, args)

    name = f"precomputed_roots-{'-'.join(args.roots)}"
    if args.masked:
        name += "_masked"
    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader,
        learning_rate=1.0e-4, device=args.device,
        log_image_interval=50
    )
    trainer.fit(args.n_iterations)


def check_loader(args, n=2):
    from torch_em.util.debug import check_loader
    patch_shape = (32, 384, 384)
    loader = get_loader("train", patch_shape, args)
    check_loader(loader, n)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper(require_input=False)
    parser.add_argument("-r", "--roots", type=str, nargs="+", required=True)
    parser.add_argument("-m", "--masked", type=int, default=1)
    args = parser.parse_args()
    if args.check:
        check_loader(args)
    else:
        train_precomuted(args)
