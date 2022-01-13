import os
from glob import glob

import numpy as np
import torch_em
from torch_em.model import AnisotropicUNet
import torch_em.shallow2deep as shallow2deep


def get_pseudolabel_loader(args, split):
    patch_shape = (32, 256, 256)

    paths = glob(os.path.join(args.input, "*.h5"))
    paths.sort()

    if split == "train":
        n_samples = 1000
        rois = [np.s_[:75, :, :]] * 4
    else:
        n_samples = 25
        rois = [np.s_[75:, :, :]] * 4
    assert len(rois) == len(paths)

    rf_path = "../boundary_prediction.ilp"
    rf_config = (rf_path, 3)
    ckpt = "./checkpoints/cremi3d-v1"

    raw_transform = torch_em.transform.raw.normalize
    # 12 workers is still not full throughput here
    loader = shallow2deep.get_pseudolabel_loader(
        raw_paths=paths, raw_key="raw", checkpoint=ckpt, rf_config=rf_config,
        batch_size=args.batch_size, patch_shape=patch_shape, rois=rois,
        raw_transform=raw_transform, n_samples=n_samples, ndim=3,
        is_raw_dataset=True, shuffle=True, num_workers=12,
    )
    return loader


def train_pseudolabels(args):
    # TODO find a version scheme for names depending on args and existing versions
    name = f"pseudolabels-v{args.version}"
    model = AnisotropicUNet(in_channels=1, out_channels=1, final_activation="Sigmoid",
                            scale_factors=[[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]])

    train_loader = get_pseudolabel_loader(args, "train")
    val_loader = get_pseudolabel_loader(args, "val")

    dice_loss = torch_em.loss.DiceLoss()
    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader, loss=dice_loss, metric=dice_loss, learning_rate=1.0e-4,
        device=args.device, log_image_interval=50
    )
    trainer.fit(args.n_iterations)


def check_loader(args, n=4):
    from torch_em.util.debug import check_loader
    loader = get_pseudolabel_loader(args, "train")
    check_loader(loader, n)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper()
    parser.add_argument("--n_rfs", type=int, default=500)
    parser.add_argument("--n_threads", type=int, default=32)
    parser.add_argument("-v", "--version", type=int, required=True)
    args = parser.parse_args()
    if args.check:
        check_loader(args)
    else:
        train_pseudolabels(args)
