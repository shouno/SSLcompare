from __future__ import annotations


def build_model(method: str, **args):
    m = method.lower()

    # BaseSSLModule が受け取るもの（MAEは base_encoder を内部で vit_mae に上書き）
    base_kwargs = dict(
        base_encoder=args.get("base_encoder", "resnet50"),
        lr=args.get("lr", 0.2),
        weight_decay=args.get("weight_decay", 1e-6),
        warmup_epochs=args.get("warmup_epochs", 10),
    )

    if m == "simclr":
        from sslcompare.models.simclr import SimCLRModule

        return SimCLRModule(
            temperature=args.get("temperature", 0.2),
            **base_kwargs,
        )

    if m in {"barlow", "barlowtwins"}:
        from sslcompare.models.barlowtwins import BarlowTwinsModule

        return BarlowTwinsModule(
            lambd=args.get("lambd", 5e-4),
            **base_kwargs,
        )

    if m == "byol":
        from sslcompare.models.byol import BYOLModule

        return BYOLModule(
            ema_decay=args.get("ema_decay", 0.996),
            **base_kwargs,
        )

    if m == "simsiam":
        from sslcompare.models.simsiam import SimSiamModule

        return SimSiamModule(**base_kwargs)

    if m == "swav":
        from sslcompare.models.swav import SwAVModule

        return SwAVModule(
            n_prototypes=args.get("n_prototypes", 3000),
            temperature=args.get("temperature", 0.1),
            n_local_crops=args.get("n_local_crops", 6),
            sinkhorn_eps=args.get("sinkhorn_eps", 0.05),
            sinkhorn_iters=args.get("sinkhorn_iters", 3),
            **base_kwargs,
        )

    if m == "mae":
        from sslcompare.models.mae import MAEModule

        # MAE は configure_optimizers を独自実装(AdamW + CosineAnnealingLR)なので、
        # lr/wd のデフォルトは MAE っぽい値に倒す方が安全です :contentReference[oaicite:3]{index=3}
        mae_kwargs = dict(
            lr=args.get("lr", 1.5e-4),
            base_encoder="vit_mae",
            weight_decay=args.get("weight_decay", 0.05),
            warmup_epochs=args.get("warmup_epochs", 10),
        )

        return MAEModule(
            img_size=args.get("img_size", 224),
            patch_size=args.get("patch_size", 16),
            mask_ratio=args.get("mask_ratio", 0.75),
            **mae_kwargs,
        )

    raise ValueError(f"Unknown method: {method}")
