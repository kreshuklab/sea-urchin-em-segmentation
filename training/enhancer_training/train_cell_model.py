import os
from glob import glob

import torch_em
import torch_em.shallow2deep as shallow2deep
from torch_em.model import AnisotropicUNet


def get_filter_config():
    filters = ["gaussianSmoothing", "laplacianOfGaussian",
               "gaussianGradientMagnitude", "hessianOfGaussianEigenvalues"]
    sigmas = [
        (0.4, 1.6, 1.6),
        (0.8, 3.5, 3.5),
        (1.25, 5.0, 5.0),
    ]
    filters_and_sigmas = [
        (filt, sigma) for filt in filters for sigma in sigmas
    ]
    return filters_and_sigmas


def prepare_shallow2deep_cells(args, out_folder):
    patch_shape_min = [16, 128, 128]
    patch_shape_max = [32, 256, 256]

    raw_transform = torch_em.transform.raw.normalize
    label_transform = shallow2deep.BoundaryTransform(ndim=3)
    paths = glob("./data/cells/*.n5")
    paths.sort()

    shallow2deep.prepare_shallow2deep(
        raw_paths=paths, raw_key="volumes/raw/s1",
        label_paths=paths, label_key="volumes/labels/segmentation/s1",
        patch_shape_min=patch_shape_min, patch_shape_max=patch_shape_max,
        n_forests=args.n_rfs, n_threads=args.n_threads,
        output_folder=out_folder, ndim=3,
        raw_transform=raw_transform, label_transform=label_transform,
        is_seg_dataset=True,
        filter_config=get_filter_config(),
    )


def get_cell_loader(args, split, rf_folder):
    rf_paths = glob(os.path.join(rf_folder, "*.pkl"))
    rf_paths.sort()
    patch_shape = (32, 256, 256)

    paths = glob("./data/cells/*.n5")
    paths.sort()

    if split == "train":
        n_samples = 1000
        paths = paths[:-1]
    else:
        n_samples = 50
        paths = paths[-1:]

    raw_transform = torch_em.transform.raw.normalize
    label_transform = torch_em.transform.BoundaryTransform(ndim=3)
    loader = shallow2deep.get_shallow2deep_loader(
        raw_paths=paths, raw_key="volumes/raw/s1",
        label_paths=paths, label_key="volumes/labels/segmentation/s1",
        rf_paths=rf_paths,
        batch_size=args.batch_size, patch_shape=patch_shape,
        raw_transform=raw_transform, label_transform=label_transform,
        n_samples=n_samples, ndim=3, is_seg_dataset=True, shuffle=True,
        num_workers=24, filter_config=get_filter_config(),
    )
    return loader


def train_shallow2deep(args):
    name = f"cells-v{args.version}"

    # check if we need to train the rfs for preparation
    rf_folder = os.path.join("checkpoints", name, "rfs")
    have_rfs = len(glob(os.path.join(rf_folder, "*.pkl"))) == args.n_rfs
    if not have_rfs:
        prepare_shallow2deep_cells(args, rf_folder)
    assert os.path.exists(rf_folder)

    model = AnisotropicUNet(in_channels=1, out_channels=1, final_activation="Sigmoid",
                            scale_factors=[[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]])

    train_loader = get_cell_loader(args, "train", rf_folder)
    val_loader = get_cell_loader(args, "val", rf_folder)

    dice_loss = torch_em.loss.DiceLoss()
    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader, loss=dice_loss, metric=dice_loss, learning_rate=1.0e-4,
        device=args.device, log_image_interval=50
    )
    trainer.fit(args.n_iterations)


def check_loader(args, n=4):
    from torch_em.util.debug import check_loader
    loader = get_cell_loader(args, "train", "./checkpoints/cells/rfs")
    check_loader(loader, n)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper(require_input=False)
    parser.add_argument("--n_rfs", type=int, default=500)
    parser.add_argument("--n_threads", type=int, default=32)
    parser.add_argument("-v", "--version", type=int, required=True)
    args = parser.parse_args()
    if args.check:
        check_loader(args)
    else:
        train_shallow2deep(args)
