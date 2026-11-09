"""Single-device, clean-reference-guided AWMM adaptation for P2Fusion.

Run --smoke-test first. Synthetic teachers are deliberately confined to that
mode: it does not read data, save training weights, or stand in for real
SegFormer/BRISQUE supervision.
"""

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path


ENCODER_NAMES = ("conv_first", "patch_embed", "rstb", "norm_Ex_ir")
NEW_PREFIXES = ("fdn_", "dwam_", "clean_reconstructor_", "residual_reconstructor_")
IMAGE_KEYS = ("ir", "vis", "clean_ir", "clean_vis")


def audit_split_manifests(train_path, val_path):
    """Refuse leaked or incorrectly assigned manifests before loading pixels."""
    documents = []
    for path, expected in ((train_path, "train"), (val_path, "val")):
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if document.get("schema_version") != 1 or document.get("split") != expected:
            raise ValueError("Expected schema_version=1 {} manifest: {}".format(expected, path))
        if not document.get("records"):
            raise ValueError("Empty {} manifest: {}".format(expected, path))
        for row in document["records"]:
            missing = {"id", "group", "clean_ir_sha256", "clean_vis_sha256"} - set(row)
            if missing:
                raise ValueError("Manifest record lacks leakage audit keys: {}".format(sorted(missing)))
        documents.append(document)
    train_rows, val_rows = (item["records"] for item in documents)
    for key in ("id", "group", "clean_ir_sha256", "clean_vis_sha256"):
        overlap = {row[key] for row in train_rows} & {row[key] for row in val_rows}
        if overlap:
            raise ValueError("Train/validation leakage in {}: {} overlapping values".format(key, len(overlap)))
    return documents


def set_encoder_frozen(model, frozen):
    """Only the original shared feature encoder is frozen during warm-up."""
    for name in ENCODER_NAMES:
        module = getattr(model, name)
        for parameter in module.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            module.eval()
        elif model.training:
            module.train()


def optimizer_groups(model, learning_rate, base_multiplier):
    if learning_rate <= 0 or not 0 < base_multiplier <= 1:
        raise ValueError("lr must be positive and base_lr_multiplier must be in (0, 1]")
    original, added = [], []
    # Include frozen encoder parameters now, so they can be unfrozen later
    # without changing the optimizer's state or parameter ordering.
    for name, parameter in model.named_parameters():
        (added if name.startswith(NEW_PREFIXES) else original).append(parameter)
    return [
        {"params": original, "lr": learning_rate * base_multiplier, "name": "original"},
        {"params": added, "lr": learning_rate, "name": "adaptation"},
    ]


def capture_rng():
    import numpy as np
    import torch
    result = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_save(payload, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def resume_semantics(config):
    """Settings that must not change underneath a resumed best-val score/run."""
    train = config["train"]
    return {
        "network": config["network"], "loss": config["loss"], "teachers": config["teachers"],
        "crop_size": config["dataset"].get("crop_size", 256),
        "train": {key: train.get(key) for key in (
            "batch_size", "lr", "base_lr_multiplier", "freeze_encoder_steps",
            "milestones", "gamma", "amp", "seed", "weight_decay", "grad_clip",
            "validation_batch_limit")},
    }


def load_training_state(path, model, optimizer, scheduler, scaler, digest, expected_config=None):
    """Restore an actual adaptation checkpoint, never an original-only weight file."""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "scheduler", "scaler", "epoch", "batch_in_epoch",
                "step", "best_val", "manifest_digest", "rng"}
    if not isinstance(checkpoint, dict) or required - set(checkpoint):
        raise ValueError("--resume requires a complete AWMM training checkpoint; use --init for source weights")
    if checkpoint["manifest_digest"] != digest:
        raise ValueError("Resume manifests differ from the saved training/validation manifests")
    if expected_config is not None:
        if "config" not in checkpoint or resume_semantics(checkpoint["config"]) != resume_semantics(expected_config):
            raise ValueError("Resume configuration changes network/loss/teachers/training or validation protocol; "
                             "use the saved configuration, or start a new run instead")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    # Move optimizer tensors onto the parameter device (checkpoint was read on CPU).
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value) and key != "step":
                state[key] = value.to(parameter.device)
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng(checkpoint["rng"])
    return checkpoint


def seed_worker(_):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if key in IMAGE_KEYS else value
            for key, value in batch.items()}


def calculate_losses(model, criterion, teachers, batch, amp_enabled):
    import torch
    with torch.no_grad():
        targets = teachers(batch["clean_ir"], batch["clean_vis"])
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        outputs = model(batch["ir"], batch["vis"], return_aux=True)
        clean_aux = model.encode_aux(batch["clean_ir"], batch["clean_vis"])
    # The loss module explicitly promotes to float32, including BCE under AMP.
    return criterion(outputs, clean_aux, batch, targets)


def finite_gradient_summary(model):
    import torch
    summary = {}
    for prefix in ("fdn_ir", "fdn_vis", "dwam_ir", "dwam_vis",
                   "clean_reconstructor_ir", "residual_reconstructor_ir",
                   "clean_reconstructor_vis", "residual_reconstructor_vis"):
        values = [p.grad for name, p in model.named_parameters()
                  if name.startswith(prefix) and p.grad is not None]
        if not values:
            raise RuntimeError("No gradients reached {}".format(prefix))
        if any(not torch.isfinite(value).all() for value in values):
            raise FloatingPointError("Nonfinite gradients in {}".format(prefix))
        norm = sum(float(value.detach().float().square().sum()) for value in values) ** 0.5
        if not norm > 0:
            raise RuntimeError("All gradients are zero in {}".format(prefix))
        summary[prefix] = norm
    return summary


def validate(model, criterion, teachers, loader, device, amp_enabled, max_batches=0):
    import torch
    previous_training = model.training
    model.eval()
    totals, count = {}, 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            batch = move_batch(batch, device)
            losses = calculate_losses(model, criterion, teachers, batch, amp_enabled)
            size = batch["ir"].shape[0]
            for key, value in losses.items():
                number = float(value.detach())
                if not math.isfinite(number):
                    raise FloatingPointError("Nonfinite validation loss: {}".format(key))
                totals[key] = totals.get(key, 0.0) + number * size
            count += size
    if not count:
        raise ValueError("Validation loader produced no samples")
    model.train(previous_training)
    return {key: value / count for key, value in totals.items()}


def append_log(path, event):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def synthetic_smoke(config, device):
    """One genuine model/loss backward pass with explicitly synthetic targets."""
    import torch
    from awmm.losses import AWMMLoss
    from awmm.runtime import make_model, seed_everything
    seed_everything(config["train"].get("seed", 3407))
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = make_model(config["network"]).to(device).train()
    loss_options = config["loss"]
    criterion = AWMMLoss(device=device, **loss_options)
    clean_ir = torch.rand(1, 3, 64, 64, device=device)
    clean_vis = torch.rand(1, 3, 64, 64, device=device)
    batch = {"clean_ir": clean_ir, "clean_vis": clean_vis,
             "ir": (clean_ir * 0.8 + 0.1).clamp(0, 1),
             "vis": (clean_vis * 0.7 + 0.2).clamp(0, 1)}
    # These labels only verify wiring; no teacher quality is claimed.
    targets = {"ir": clean_ir.mean(1, keepdim=True),
               "vis": clean_vis.new_full((1, 1), 0.8)}
    outputs = model(batch["ir"], batch["vis"], return_aux=True)
    auxiliary = model.encode_aux(clean_ir, clean_vis)
    losses = criterion(outputs, auxiliary, batch, targets)
    if any(not torch.isfinite(value).all() for value in losses.values()):
        raise FloatingPointError("Synthetic loss is not finite")
    losses["total"].backward()
    report = {"mode": "synthetic_smoke_only", "device": str(device),
              "fused_shape": list(outputs["fused"].shape),
              "losses": {key: float(value.detach()) for key, value in losses.items()},
              "gradient_norms": finite_gradient_summary(model),
              "checkpoint_saved": False}
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/awmm.json")
    parser.add_argument("--data-root")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--run-dir")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--init", help="Original P2Fusion checkpoint; only new modules may be missing")
    source.add_argument("--resume", help="Full AWMM training checkpoint, including optimizer and manifests")
    source.add_argument("--from-scratch", action="store_true", help="Explicitly train without source weights")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument("--max-steps", type=int, default=0, help="Absolute optimizer step limit; 0 uses epochs")
    parser.add_argument("--val-batches", type=int, default=0, help="0 validates the complete validation manifest")
    parser.add_argument("--smoke-test", action="store_true", help="Synthetic wiring test; no data/teachers/checkpoints")
    args = parser.parse_args(argv)
    if args.max_steps < 0 or args.val_batches < 0:
        parser.error("--max-steps and --val-batches must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    from torch.utils.data import DataLoader
    from awmm.data import AWMMFourImageDataset
    from awmm.losses import AWMMLoss
    from awmm.runtime import load_config, make_model, manifest_digest, seed_everything, build_teachers

    config = copy.deepcopy(load_config(args.config))
    for argument, section, key in ((args.data_root, "dataset", "root"),
                                    (args.manifest_dir, "dataset", "manifest_dir")):
        if argument:
            config[section][key] = argument
    if args.run_dir:
        config["run_dir"] = args.run_dir
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU")
    if args.smoke_test:
        return synthetic_smoke(config, device)

    train_options = config["train"]
    train_options["validation_batch_limit"] = args.val_batches
    seed = int(train_options.get("seed", 3407))
    seed_everything(seed)
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    crop_size = int(config["dataset"].get("crop_size", 256))
    if crop_size < 32 or crop_size % 32:
        raise ValueError("Training crop_size must be a positive multiple of 32")
    manifest_dir = Path(config["dataset"]["manifest_dir"])
    train_manifest, val_manifest = manifest_dir / "train.json", manifest_dir / "val.json"
    audit_split_manifests(train_manifest, val_manifest)
    digest = manifest_digest([train_manifest, val_manifest])
    train_dataset = AWMMFourImageDataset(config["dataset"]["root"], train_manifest,
                                        crop_size=crop_size, training=True)
    val_dataset = AWMMFourImageDataset(config["dataset"]["root"], val_manifest,
                                      crop_size=crop_size, training=False)
    batch_size = int(train_options.get("batch_size", 2))
    workers = int(train_options.get("num_workers", 4))
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    loader_arguments = dict(batch_size=batch_size, num_workers=workers,
                            pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                            drop_last=False, persistent_workers=False)
    # This generator is reset by epoch: shuffling and skipped batch positions
    # are reproducible. Worker crop randomness after mid-epoch resume is not
    # promised to be bit-for-bit identical to uninterrupted multi-worker runs.
    generator = torch.Generator()
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_arguments)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_arguments)
    model = make_model(config["network"]).to(device)
    initial = args.init or (None if args.from_scratch or args.resume else config.get("init_checkpoint"))
    if not args.resume and not initial and not args.from_scratch:
        raise ValueError("Provide --init /path/to/source.pth or explicitly use --from-scratch")
    if initial:
        print("Source checkpoint load report:", model.load_initial_checkpoint(initial), flush=True)
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, float(train_options.get("lr", 1e-4)),
                         float(train_options.get("base_lr_multiplier", 0.1))),
        weight_decay=float(train_options.get("weight_decay", 0.0)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=train_options.get("milestones", [3000, 6000, 9000, 12000, 15000]),
        gamma=float(train_options.get("gamma", 0.5)))
    amp_enabled = bool(train_options.get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    criterion = AWMMLoss(device=device, **config["loss"])
    step, epoch_start, batch_start, best_val = 0, 0, 0, float("inf")
    if args.resume:
        state = load_training_state(args.resume, model, optimizer, scheduler, scaler, digest,
                                    expected_config=config)
        step, epoch_start, batch_start = int(state["step"]), int(state["epoch"]), int(state["batch_in_epoch"])
        best_val = float(state["best_val"])
        print("Resumed at step {}, epoch {}, batch {}".format(step, epoch_start, batch_start), flush=True)
    teachers = build_teachers(config, device)
    if args.resume:
        # Constructing frozen teachers initializes parameters before loading
        # their weights; keep that initialization from consuming resumed RNG.
        restore_rng(state["rng"])
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and ((run_dir / "last.pt").exists() or (run_dir / "best.pt").exists()):
        raise FileExistsError("Run already contains checkpoints; use --resume or a new --run-dir")
    (run_dir / "config_resolved.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    log_path = run_dir / "metrics.jsonl"
    append_log(log_path, {"event": "start", "device": str(device), "torch": torch.__version__,
                          "cuda_runtime": torch.version.cuda, "amp": amp_enabled,
                          "train_samples": len(train_dataset), "val_samples": len(val_dataset),
                          "validation": "deterministic_center_crop", "val_batch_limit": args.val_batches,
                          "resume": args.resume, "step": step})
    writer = None
    if train_options.get("tensorboard", False):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run_dir / "tensorboard"))
    freeze_steps = int(train_options.get("freeze_encoder_steps", 500))
    log_every = max(1, int(train_options.get("log_every", 20)))
    val_every = max(1, int(train_options.get("val_every", 1000)))
    save_every = max(1, int(train_options.get("save_every", 1000)))
    epochs = int(train_options.get("epochs", 100))
    if epochs < 1:
        raise ValueError("epochs must be positive")
    start_time, last_validation_step = time.time(), -1

    def save(name, next_epoch, next_batch):
        atomic_save({"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": next_epoch, "batch_in_epoch": next_batch, "step": step,
                     "best_val": best_val, "config": config, "manifest_digest": digest,
                     "rng": capture_rng()}, run_dir / name)

    def evaluate(next_epoch, next_batch):
        nonlocal best_val, last_validation_step
        result = validate(model, criterion, teachers, val_loader, device, amp_enabled, args.val_batches)
        append_log(log_path, dict(event="validation", step=step, **result))
        if writer:
            for key, value in result.items():
                writer.add_scalar("validation/" + key, value, step)
        improved = result["total"] < best_val
        if improved:
            best_val = result["total"]
            save("best.pt", next_epoch, next_batch)
        last_validation_step = step
        set_encoder_frozen(model, step < freeze_steps)

    try:
        if args.max_steps and step >= args.max_steps:
            print("Absolute --max-steps already reached; no optimizer steps executed", flush=True)
            return
        for epoch in range(epoch_start, epochs):
            generator.manual_seed(seed + epoch)
            model.train()
            set_encoder_frozen(model, step < freeze_steps)
            for batch_index, batch in enumerate(train_loader):
                if epoch == epoch_start and batch_index < batch_start:
                    continue
                if step >= freeze_steps:
                    set_encoder_frozen(model, False)
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                losses = calculate_losses(model, criterion, teachers, batch, amp_enabled)
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError("Nonfinite training loss at step {}".format(step + 1))
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                # Fail loudly rather than silently retaining a nonfinite model.
                finite_flags = [torch.isfinite(parameter.grad).all()
                                for parameter in model.parameters() if parameter.grad is not None]
                nonfinite_gradients = bool(finite_flags) and not bool(torch.stack(finite_flags).all())
                if nonfinite_gradients:
                    if not amp_enabled:
                        raise FloatingPointError("Nonfinite training gradient at step {}".format(step + 1))
                    # GradScaler recorded this overflow during unscale_. Its
                    # step skips the optimizer and update lowers the scale.
                    # Do not advance scheduler or optimizer-step counters.
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    append_log(log_path, {"event": "amp_overflow_step_skipped", "step": step,
                                          "epoch": epoch, "batch": batch_index,
                                          "new_scale": scaler.get_scale()})
                    continue
                clipping = float(train_options.get("grad_clip", 1.0))
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                next_epoch = epoch + (batch_index + 1 == len(train_loader))
                next_batch = 0 if next_epoch > epoch else batch_index + 1
                if step == 1 or step % log_every == 0:
                    numbers = {key: float(value.detach()) for key, value in losses.items()}
                    append_log(log_path, dict(event="training", step=step, epoch=epoch,
                                             lr=[group["lr"] for group in optimizer.param_groups],
                                             elapsed_seconds=round(time.time() - start_time, 2), **numbers))
                    if writer:
                        for key, value in numbers.items():
                            writer.add_scalar("training/" + key, value, step)
                stopping = bool(args.max_steps and step >= args.max_steps)
                if step % val_every == 0 or stopping:
                    evaluate(next_epoch, next_batch)
                if step % save_every == 0 or stopping:
                    save("last.pt", next_epoch, next_batch)
                if stopping:
                    return
            batch_start = 0
            if last_validation_step != step:
                evaluate(epoch + 1, 0)
            save("last.pt", epoch + 1, 0)
    finally:
        if writer:
            writer.close()


if __name__ == "__main__":
    main()
