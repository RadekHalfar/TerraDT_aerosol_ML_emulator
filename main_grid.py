import argparse
import itertools
import sys
from datetime import datetime

import torch
import torch.nn as nn

from train_time_approach import train_model
from model_UNet3D_temporal import UNet3DTemporal
from model_ConvLSTM import KappaPredictorConvLSTM
from model_UNet_test import UNet3D


def make_optimizer(model, name: str, lr: float, weight_decay: float):
    name = (name or "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def make_loss(loss_name: str):
    loss_name = (loss_name or "mse").lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "smoothl1":
        # beta tuned for regression stability; can be overridden in future
        return nn.SmoothL1Loss(beta=0.1)
    if loss_name == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def generate_unet_temporal_grid(preset: str):
    # Coarse but strong defaults
    if preset == "quick":
        combos = [
            dict(temporal_mode="attn", base_ch=32, depth=3, use_residual=True, use_se=True, attn_heads=4, fuse_multiscale=False, seq_len=5,
                 lr=3e-4, weight_decay=1e-4, batch_size=1, loss="mse"),
            dict(temporal_mode="mean", base_ch=32, depth=3, use_residual=True, use_se=True, attn_heads=None, fuse_multiscale=True, seq_len=5,
                 lr=3e-4, weight_decay=1e-4, batch_size=1, loss="mse"),
            dict(temporal_mode="last", base_ch=32, depth=3, use_residual=True, use_se=False, attn_heads=None, fuse_multiscale=False, seq_len=1,
                 lr=1e-3, weight_decay=0.0, batch_size=2, loss="mse"),
        ]
        return combos

    # full preset: cartesian around promising areas
    modes = ["attn", "mean", "last"]
    base_ch_list = [32, 48]
    depth_list = [2, 3]
    residual_list = [True]
    se_list = [True, False]
    fuse_list = [False, True]
    seq_lens = {"attn": [3, 5], "mean": [3, 5], "last": [1]}
    attn_heads_map = {"attn": [2, 4]}
    lrs = [3e-4, 1e-4]
    wds = [1e-4, 0.0]
    batch_sizes = [1]
    losses = ["mse", "smoothl1"]

    combos = []
    for mode in modes:
        for base_ch, depth, use_residual, use_se in itertools.product(base_ch_list, depth_list, residual_list, se_list):
            for fuse_multiscale in fuse_list:
                for seq_len in seq_lens[mode]:
                    for lr in lrs:
                        for wd in wds:
                            for bs in batch_sizes:
                                for loss in losses:
                                    cfg = dict(
                                        temporal_mode=mode,
                                        base_ch=base_ch,
                                        depth=depth,
                                        use_residual=use_residual,
                                        use_se=use_se,
                                        fuse_multiscale=fuse_multiscale,
                                        seq_len=seq_len,
                                        lr=lr,
                                        weight_decay=wd,
                                        batch_size=bs,
                                        loss=loss,
                                    )
                                    if mode == "attn":
                                        for h in attn_heads_map["attn"]:
                                            c = dict(cfg)
                                            c["attn_heads"] = h
                                            combos.append(c)
                                    else:
                                        cfg["attn_heads"] = None
                                        combos.append(cfg)
    # Filter impractical: fuse_multiscale only meaningful when seq_len>1
    combos = [c for c in combos if (c["seq_len"] > 1 or c["fuse_multiscale"] is False)]
    return combos


def generate_convlstm_grid(preset: str):
    if preset == "quick":
        return [
            dict(hidden_channels=64, num_layers=2, kernel_size=3, seq_len=5, lr=3e-4, weight_decay=1e-4, batch_size=1, loss="mse"),
            dict(hidden_channels=64, num_layers=1, kernel_size=3, seq_len=3, lr=3e-4, weight_decay=1e-4, batch_size=1, loss="mse"),
        ]

    hidden_list = [32, 64, 96]
    layers_list = [1, 2]
    ksize_list = [3, 5]
    seq_list = [3, 5, 7]
    lrs = [3e-4, 1e-4]
    wds = [1e-4, 0.0]
    batch_sizes = [1]
    losses = ["mse", "smoothl1"]

    combos = []
    for h, L, k, seq_len, lr, wd, bs, loss in itertools.product(
        hidden_list, layers_list, ksize_list, seq_list, lrs, wds, batch_sizes, losses
    ):
        combos.append(dict(hidden_channels=h, num_layers=L, kernel_size=k, seq_len=seq_len,
                           lr=lr, weight_decay=wd, batch_size=bs, loss=loss))
    return combos


def generate_baselines_grid(preset: str):
    # Keep small; baselines are for sanity
    return [
        dict(model="UNet3D", bilinear=True, seq_len=1, lr=3e-4, weight_decay=1e-4, batch_size=1, loss="mse"),
        dict(model="UNet3D", bilinear=False, seq_len=1, lr=1e-3, weight_decay=0.0, batch_size=1, loss="mse"),
        dict(model="CNN", seq_len=1, lr=1e-3, weight_decay=0.0, batch_size=2, loss="mse"),
    ]


def run_config(nc_path: str,
               model_kind: str,
               cfg: dict,
               device: str,
               epochs: int,
               optimizer_name: str,
               experiment_name: str,
               n_splits: int,
               gap: int,
               show_fold_plot: bool,
               plot: bool):
    try:
        if model_kind == "unet_temporal":
            model = UNet3DTemporal(
                in_channels=12,
                out_channels=2,
                base_ch=cfg["base_ch"],
                depth=cfg["depth"],
                temporal_mode=cfg["temporal_mode"],
                attn_heads=(cfg.get("attn_heads") or 4),
                use_residual=cfg["use_residual"],
                use_se=cfg["use_se"],
                fuse_multiscale=cfg["fuse_multiscale"],
            ).to(device)
        elif model_kind == "convlstm":
            model = KappaPredictorConvLSTM(
                in_channels=12,
                hidden_channels=cfg["hidden_channels"],
                num_layers=cfg["num_layers"],
                kernel_size=cfg["kernel_size"],
                out_channels=2,
            ).to(device)
        elif model_kind == "unet3d_base":
            model = UNet3D(in_channels=12, n_classes=2, bilinear=cfg.get("bilinear", True)).to(device)
        elif model_kind == "cnn_base":
            from model_CNN_test import KappaPredictorCNN
            model = KappaPredictorCNN(in_channels=12).to(device)
        else:
            raise ValueError(f"Unknown model_kind {model_kind}")

        lr = float(cfg["lr"]) if "lr" in cfg else 3e-4
        wd = float(cfg.get("weight_decay", 0.0))
        loss_fn = make_loss(cfg.get("loss", "mse"))
        optimizer = make_optimizer(model, optimizer_name, lr, wd)
        seq_len = int(cfg.get("seq_len", 1))

        print(f"\n=== Running {model_kind} | cfg={cfg} ===")
        train_model(
            nc_file_path=nc_path,
            epochs=epochs,
            batch_size=int(cfg.get("batch_size", 1)),
            lr=lr,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            plot=plot,
            experiment_name=experiment_name,
            n_splits=n_splits,
            gap=gap,
            show_fold_plot=show_fold_plot,
            seq_len=seq_len,
        )
    except RuntimeError as e:
        # Handle OOM gracefully and continue grid
        if "out of memory" in str(e).lower():
            print(f"[OOM] Skipping config due to CUDA OOM: {cfg}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            raise


def main():
    parser = argparse.ArgumentParser(description="Run grid of model configs with MLflow logging.")
    parser.add_argument("--nc", default="hamlite_sample_data_filtered.nc", help="Path to NetCDF dataset")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for all runs")
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"])
    parser.add_argument("--which", default="all", choices=["all", "unet_temporal", "convlstm", "baselines"])
    parser.add_argument("--preset", default="quick", choices=["quick", "full"], help="Grid size")
    parser.add_argument("--experiment", default=None, help="MLflow experiment base name")
    parser.add_argument("--n-splits", type=int, default=1, help="TimeSeriesSplit folds (>=2 for CV)")
    parser.add_argument("--gap", type=int, default=0, help="Gap between train and validation in TimeSeriesSplit")
    parser.add_argument("--plot", action="store_true", help="Save loss plots")
    parser.add_argument("--show-fold-plot", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    device = args.device
    experiment = args.experiment or f"GridSearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    grids = []
    if args.which in ("all", "unet_temporal"):
        grids.append(("unet_temporal", generate_unet_temporal_grid(args.preset)))
    if args.which in ("all", "convlstm"):
        grids.append(("convlstm", generate_convlstm_grid(args.preset)))
    if args.which in ("all", "baselines"):
        grids.append(("unet3d_base", [c for c in generate_baselines_grid(args.preset) if c.get("model") == "UNet3D"]))
        grids.append(("cnn_base", [c for c in generate_baselines_grid(args.preset) if c.get("model") == "CNN"]))

    # Remove duplicates for baselines list calls
    # Execute
    total = sum(len(cfgs) for _, cfgs in grids)
    print(f"Planned runs: {total} (preset={args.preset}, which={args.which})")

    run_idx = 0
    for model_kind, cfgs in grids:
        for cfg in cfgs:
            run_idx += 1
            if args.batch_size is not None:
                cfg = dict(cfg)
                cfg["batch_size"] = int(args.batch_size)
            print(f"\n[Run {run_idx}/{total}] {model_kind} -> {cfg}")
            run_config(
                nc_path=args.nc,
                model_kind=model_kind,
                cfg=cfg,
                device=device,
                epochs=args.epochs,
                optimizer_name=args.optimizer,
                experiment_name=experiment,
                n_splits=args.n_splits,
                gap=args.gap,
                show_fold_plot=args.show_fold_plot,
                plot=args.plot,
            )

    print("\nAll grid runs finished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
