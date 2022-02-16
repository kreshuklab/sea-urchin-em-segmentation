import torch_em
from torch_em.model import UNet3d
from torch_em.util import parser_helper


# TODO check whether the current model does anything (training looks quite weird)
def get_loader(args, patch_shape):
    path = "./data/neuropil/data.n5"
    sampler = torch_em.data.MinForegroundSampler(0.05)
    return torch_em.default_segmentation_loader(path, "raw", path, "labels",
                                                args.batch_size, patch_shape,
                                                sampler=sampler, shuffle=True,
                                                num_workers=4*args.batch_size)


def train_neuropil(args):
    model = UNet3d(in_channels=1, out_channels=1, final_activation="Sigmoid",
                   depth=5, initial_features=16)
    patch_shape = [32, 256, 256]
    train_loader = get_loader(args, patch_shape=patch_shape)
    val_loader = get_loader(args, patch_shape=patch_shape)
    name = "neuropil-model"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=0.5e-5,
        mixed_precision=True,
        log_image_interval=50,
        device=args.device
    )
    trainer.fit(args.n_iterations)


if __name__ == "__main__":
    parser = parser_helper(require_input=False)
    args = parser.parse_args()
    train_neuropil(args)
