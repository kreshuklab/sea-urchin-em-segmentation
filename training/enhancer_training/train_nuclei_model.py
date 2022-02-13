import os
from glob import glob

import torch_em
import torch_em.shallow2deep as shallow2deep
from torch_em.model import AnisotropicUNet


def get_filter_config():
    filters = ["gaussianSmoothing", "laplacianOfGaussian",
               "gaussianGradientMagnitude", "hessianOfGaussianEigenvalues"]
    sigmas = [1.6, 3.5, 5.0]
    filters_and_sigmas = [
        (filt, sigma) for filt in filters for sigma in sigmas
    ]
    return filters_and_sigmas


def to_binary(seg):
    return (seg > 0).astype("float32")


def prepare_shallow2deep_nuclei(args, out_folder):
    patch_shape_min = [32, 128, 128]
    patch_shape_max = [64, 256, 256]

    raw_transform = torch_em.transform.raw.normalize
    paths = glob("/scratch/pape/platy/nuclei/*.h5")
    paths.sort()

    shallow2deep.prepare_shallow2deep(
        raw_paths=paths, raw_key="volumes/raw",
        label_paths=paths, label_key="volumes/labels/nucleus_instance_labels",
        patch_shape_min=patch_shape_min, patch_shape_max=patch_shape_max,
        n_forests=args.n_rfs, n_threads=args.n_threads,
        output_folder=out_folder, ndim=3,
        raw_transform=raw_transform, label_transform=to_binary,
        is_seg_dataset=True,
        filter_config=get_filter_config(),
        balance_labels=False
    )


def get_nucleus_loader(args, split, rf_folder):
    rf_paths = glob(os.path.join(rf_folder, "*.pkl"))
    rf_paths.sort()
    patch_shape = (64, 192, 192)

    paths = glob("/scratch/pape/platy/nuclei/*.h5")
    paths.sort()

    if split == "train":
        n_samples = 1000
        paths = paths[:-1]
    else:
        n_samples = 25
        paths = paths[-1:]

    raw_transform = torch_em.transform.raw.normalize
    loader = shallow2deep.get_shallow2deep_loader(
        raw_paths=paths, raw_key="volumes/raw",
        label_paths=paths, label_key="volumes/labels/nucleus_instance_labels",
        rf_paths=rf_paths,
        batch_size=args.batch_size, patch_shape=patch_shape,
        raw_transform=raw_transform, label_transform=to_binary,
        n_samples=n_samples, ndim=3, is_seg_dataset=True, shuffle=True,
        num_workers=24, filter_config=get_filter_config(),
    )
    return loader


def train_shallow2deep(args):
    name = f"nuclei-v{args.version}"

    # check if we need to train the rfs for preparation
    rf_folder = os.path.join("checkpoints", name, "rfs")
    have_rfs = len(glob(os.path.join(rf_folder, "*.pkl"))) == args.n_rfs
    if not have_rfs:
        prepare_shallow2deep_nuclei(args, rf_folder)
    assert os.path.exists(rf_folder)

    model = AnisotropicUNet(in_channels=1, out_channels=1, final_activation="Sigmoid",
                            scale_factors=[[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]])

    train_loader = get_nucleus_loader(args, "train", rf_folder)
    val_loader = get_nucleus_loader(args, "val", rf_folder)

    dice_loss = torch_em.loss.DiceLoss()
    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader, loss=dice_loss, metric=dice_loss, learning_rate=1.0e-4,
        device=args.device, log_image_interval=50
    )
    trainer.fit(args.n_iterations)


def check_loader(args, n=4):
    from torch_em.util.debug import check_loader
    loader = get_nucleus_loader(args, "train", "./checkpoints/nuclei/rfs")
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
