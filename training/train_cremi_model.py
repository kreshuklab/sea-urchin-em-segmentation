import os
from glob import glob
from functools import partial

import numpy as np
import torch_em
import torch_em.shallow2deep as shallow2deep
from torch_em.model import AnisotropicUNet

try:
    import fastfilters as filter_impl
except ImportError:
    import vigra.filters as filter_impl


def get_filter_config():
    filters = [filter_impl.gaussianSmoothing,
               filter_impl.laplacianOfGaussian,
               filter_impl.gaussianGradientMagnitude,
               filter_impl.hessianOfGaussianEigenvalues,
               filter_impl.structureTensorEigenvalues]
    sigmas = [
        (0.35, 0.7, 0.7),
        (0.8, 1.6, 1.6),
        (1.75, 3.5, 3.5),
        (2.5, 5.0, 5.0),
    ]
    filters_and_sigmas = [
        (filt, sigma) if i != len(filters) - 1 else (partial(filt, outerScale=tuple(2 * sig for sig in sigma)), sigma)
        for i, filt in enumerate(filters) for sigma in sigmas
    ]
    return filters_and_sigmas


def prepare_shallow2deep_cremi(args, out_folder):
    patch_shape_min = [16, 256, 256]
    patch_shape_max = [32, 384, 384]

    raw_transform = torch_em.transform.raw.normalize
    label_transform = shallow2deep.BoundaryTransform(ndim=3)
    paths = glob(os.path.join(args.input, "*.h5"))
    paths.sort()

    shallow2deep.prepare_shallow2deep(
        raw_paths=paths, raw_key="volumes/raw",
        label_paths=paths, label_key="volumes/labels/neuron_ids",
        patch_shape_min=patch_shape_min, patch_shape_max=patch_shape_max,
        n_forests=args.n_rfs, n_threads=args.n_threads,
        output_folder=out_folder, ndim=3,
        raw_transform=raw_transform, label_transform=label_transform,
        is_seg_dataset=True,
        filter_config=get_filter_config(),
    )


def get_cremi_loader(args, split, rf_folder):
    rf_paths = glob(os.path.join(rf_folder, "*.pkl"))
    rf_paths.sort()
    patch_shape = (32, 256, 256)

    paths = glob(os.path.join(args.input, "*.h5"))
    paths.sort()

    if split == "train":
        n_samples = 1000
        rois = np.s_[:75, :, :]
        assert len(rois) == len(paths)
    else:
        n_samples = 25
        rois = np.s_[75:, :, :]
        paths = paths[-1:]
        assert len(paths) == 1

    raw_transform = torch_em.transform.raw.normalize
    label_transform = torch_em.transform.BoundaryTransform(ndim=3)
    loader = shallow2deep.get_shallow2deep_loader(
        raw_paths=paths, raw_key="volumes/raw",
        label_paths=paths, label_key="volumes/labels/neuron_ids",
        rf_paths=rf_paths,
        batch_size=args.batch_size, patch_shape=patch_shape, rois=rois,
        raw_transform=raw_transform, label_transform=label_transform,
        n_samples=n_samples, ndim=3, is_seg_dataset=True, shuffle=True,
        num_workers=8, filter_config=get_filter_config(),
    )
    return loader


def train_shallow2deep(args):
    # TODO find a version scheme for names depending on args and existing versions
    name = "cremi3d"

    # check if we need to train the rfs for preparation
    rf_folder = os.path.join("checkpoints", name, "rfs")
    have_rfs = len(glob(os.path.join(rf_folder, "*.pkl"))) == args.n_rfs
    if not have_rfs:
        prepare_shallow2deep_cremi(args, rf_folder)
    assert os.path.exists(rf_folder)

    model = AnisotropicUNet(in_channels=1, out_channels=1, final_activation="Sigmoid",
                            scale_factors=[[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]])

    train_loader = get_cremi_loader(args, "train", rf_folder)
    val_loader = get_cremi_loader(args, "val", rf_folder)

    dice_loss = torch_em.loss.DiceLoss()
    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader, loss=dice_loss, metric=dice_loss, learning_rate=1.0e-4,
        device=args.device, log_image_interval=50
    )
    trainer.fit(args.n_iterations)


def check_loader(args, n=4):
    from torch_em.util.debug import check_loader
    loader = get_cremi_loader(args, "train", "./checkpoints/cremi3d/rfs")
    check_loader(loader, n)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper()
    # parser.add_argument("--n_rfs", type=int, default=500)
    # parser.add_argument("--n_threads", type=int, default=32)
    parser.add_argument("--n_rfs", type=int, default=1)
    parser.add_argument("--n_threads", type=int, default=1)
    args = parser.parse_args()
    if args.check:
        check_loader(args)
    else:
        train_shallow2deep(args)
