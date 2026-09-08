from contextlib import contextmanager
import gc
import random
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from sprc_rd.training.common import rd_feature_loss


def _sample_mask_shape(
    min_side: int,
    max_side: int,
    stripe_prob: float,
    max_h: int,
    max_w: int,
) -> Tuple[int, int]:
    min_side = max(1, int(min_side))
    max_side = max(min_side, int(max_side))
    if max_side >= 2 and random.random() < float(stripe_prob):
        short_side = random.randint(min_side, min(max_side, 2))
        long_side = random.randint(max(short_side, 2), max_side)
        if random.random() < 0.5:
            bh, bw = long_side, short_side
        else:
            bh, bw = short_side, long_side
    else:
        bh = random.randint(min_side, max_side)
        bw = random.randint(min_side, max_side)
    return min(bh, max_h), min(bw, max_w)


def generate_coarse_mask(
    batch: int,
    h: int,
    w: int,
    device: torch.device,
    min_blocks: int = 1,
    max_blocks: int = 2,
    min_side: int = 1,
    max_side: int = 5,
    stripe_prob: float = 0.5,
    margin: int = 1,
    max_fraction: float = 0.30,
) -> torch.Tensor:

    mask = torch.zeros(batch, 1, h, w, device="cpu", dtype=torch.float32)
    margin = max(0, int(margin))
    min_blocks = max(1, int(min_blocks))
    max_blocks = max(min_blocks, int(max_blocks))
    for i in range(batch):
        for _ in range(random.randint(min_blocks, max_blocks)):
            bh, bw = _sample_mask_shape(min_side, max_side, stripe_prob, h, w)
            y_lo, x_lo = margin, margin
            y_hi, x_hi = h - margin - bh, w - margin - bw
            if y_hi < y_lo:
                y_lo, y_hi = 0, max(0, h - bh)
            if x_hi < x_lo:
                x_lo, x_hi = 0, max(0, w - bw)
            y = random.randint(y_lo, y_hi) if y_hi > y_lo else y_lo
            x = random.randint(x_lo, x_hi) if x_hi > x_lo else x_lo
            cand = mask[i].clone()
            cand[:, y:y + bh, x:x + bw] = 1.0
            if float(cand.mean()) <= float(max_fraction) or float(mask[i].sum()) == 0.0:
                mask[i] = cand
        if float(mask[i].sum()) == 0.0:
            mask[i, :, h // 2:h // 2 + 1, w // 2:w // 2 + 1] = 1.0
    return mask.to(device=device, non_blocking=True)


def _candidate_shifts(h: int, w: int, min_shift: int, max_shift: int, n: int):
    max_shift = max(1, min(int(max_shift), max(h, w) - 1))
    min_shift = max(1, min(int(min_shift), max_shift))
    out: List[Tuple[int, int]] = []
    attempts = 0
    while len(out) < max(1, int(n)) and attempts < max(50, 10 * int(n)):
        attempts += 1
        dy = random.randint(-max_shift, max_shift)
        dx = random.randint(-max_shift, max_shift)
        if dy == 0 and dx == 0:
            continue
        if max(abs(dy), abs(dx)) < min_shift:
            continue
        if (dy, dx) not in out:
            out.append((dy, dx))
    return out if out else [(min_shift, 0)]


@torch.no_grad()
def choose_hard_same_image_shifts(
    clean_coarse: torch.Tensor,
    coarse_mask: torch.Tensor,
    min_shift: int = 2,
    max_shift: int = 8,
    candidates: int = 4,
    max_source_overlap: float = 0.10,
) -> Tuple[List[Tuple[int, int]], List[float]]:

    b, _, h, w = clean_coarse.shape
    feat_n = F.normalize(clean_coarse.float(), dim=1, eps=1e-6)
    denom = coarse_mask.flatten(1).sum(dim=1).clamp_min(1.0)
    cand = _candidate_shifts(h, w, min_shift, max_shift, max(1, int(candidates)))

    dist_cols = []
    overlap_cols = []
    for dy, dx in cand:
        donor = torch.roll(feat_n, shifts=(dy, dx), dims=(-2, -1))
        dist_map = 1.0 - (feat_n * donor).sum(dim=1, keepdim=True)
        dist_cols.append((dist_map * coarse_mask).flatten(1).sum(dim=1) / denom)
        source_mask = torch.roll(coarse_mask, shifts=(dy, dx), dims=(-2, -1))
        overlap_cols.append((coarse_mask * source_mask).flatten(1).sum(dim=1) / denom)

    dist = torch.stack(dist_cols, dim=1)
    overlap = torch.stack(overlap_cols, dim=1)
    valid = overlap <= float(max_source_overlap)
    masked = dist.masked_fill(~valid, float("-inf"))
    idx_valid = masked.argmax(dim=1)
    idx_fallback = dist.argmax(dim=1)
    idx = torch.where(valid.any(dim=1), idx_valid, idx_fallback)
    chosen_dist = dist.gather(1, idx[:, None])[:, 0]

    shifts = [cand[int(j)] for j in idx.detach().cpu().tolist()]
    dists = [float(v) for v in chosen_dist.detach().cpu().tolist()]
    return shifts, dists


@torch.no_grad()
def corrupt_pseudo_subset_inplace(
    student_input_features: Sequence[torch.Tensor],
    clean_pseudo_targets: Sequence[torch.Tensor],
    clean_count: int,
    coarse_mask: torch.Tensor,
    shifts: Sequence[Tuple[int, int]],
    missing_probability: float = 0.50,
    missing_kernel: int = 3,
) -> Tuple[List[torch.Tensor], List[str]]:














    if len(student_input_features) != 3 or len(clean_pseudo_targets) != 3:
        raise ValueError("Expected exactly three feature scales")
    if len(shifts) != clean_pseudo_targets[0].shape[0]:
        raise ValueError("one donor shift is required per pseudo sample")
    missing_probability = min(1.0, max(0.0, float(missing_probability)))
    k = max(3, int(missing_kernel))
    if k % 2 == 0:
        k += 1

    modes = [
        "missing" if random.random() < missing_probability else "replacement"
        for _ in range(clean_pseudo_targets[0].shape[0])
    ]
    hc, wc = clean_pseudo_targets[2].shape[-2:]
    masks: List[torch.Tensor] = []

    for feat_input, clean in zip(student_input_features, clean_pseudo_targets):
        b_p, _, h, w = clean.shape
        m = F.interpolate(coarse_mask, size=(h, w), mode="nearest")
        target_slice = feat_input[clean_count:clean_count + b_p]

        
        
        pad = k // 2
        padded = F.pad(clean.float(), (pad, pad, pad, pad), mode="replicate")
        local_avg = F.avg_pool2d(padded, kernel_size=k, stride=1)
        ring_fill = (local_avg * float(k * k) - clean.float()) / float(k * k - 1)
        ring_fill = ring_fill.to(dtype=clean.dtype)

        for i, (dy_c, dx_c) in enumerate(shifts):
            if modes[i] == "missing":
                donor = ring_fill[i:i + 1]
            else:
                dy = int(round(dy_c * h / max(hc, 1)))
                dx = int(round(dx_c * w / max(wc, 1)))
                if dy_c != 0 and dy == 0:
                    dy = 1 if dy_c > 0 else -1
                if dx_c != 0 and dx == 0:
                    dx = 1 if dx_c > 0 else -1
                donor = torch.roll(clean[i:i + 1], shifts=(dy, dx), dims=(-2, -1))
            target_slice[i:i + 1].copy_(
                clean[i:i + 1] * (1.0 - m[i:i + 1]) + donor * m[i:i + 1]
            )
        masks.append(m)
    return masks, modes


@torch.no_grad()
def transplant_pseudo_subset_inplace(
    student_input_features: Sequence[torch.Tensor],
    clean_pseudo_targets: Sequence[torch.Tensor],
    clean_count: int,
    coarse_mask: torch.Tensor,
    shifts: Sequence[Tuple[int, int]],
) -> List[torch.Tensor]:

    masks, _ = corrupt_pseudo_subset_inplace(
        student_input_features, clean_pseudo_targets, clean_count, coarse_mask, shifts,
        missing_probability=0.0,
    )
    return masks


def _pseudo_count(batch_size: int, fraction: float) -> int:
    if batch_size <= 1 or fraction <= 0:
        return 0
    n = int(round(batch_size * float(fraction)))
    return max(1, min(batch_size - 1, n))


def _image_aug_count(batch_size: int, fraction: float, available_clean: int) -> int:

    if batch_size <= 0 or available_clean <= 0 or fraction <= 0:
        return 0
    n = int(round(batch_size * float(fraction)))
    return max(1, min(int(available_clean), n))


def _all_finite(tensors: Sequence[torch.Tensor]) -> bool:

    return all(bool(torch.isfinite(x).all().item()) for x in tensors)


def image_aug_progress(epoch: int, args) -> float:

    start = int(args.img_aug_start_epoch)
    if epoch < start:
        return 0.0
    ramp_epochs = max(0, int(args.img_aug_ramp_epochs))
    if ramp_epochs == 0:
        return 1.0
    return min(1.0, max(0.0, float(epoch - start) / float(ramp_epochs)))


def image_aug_schedule(epoch: int, args) -> Tuple[float, float, float, float]:













    progress = image_aug_progress(epoch, args)

    max_weight = max(0.0, float(args.img_aug_lambda_max))
    if max_weight <= 0.0 or float(args.img_aug_fraction) <= 0.0:
        weight = 0.0
    else:
        weight = max_weight * progress

    min_fraction = float(args.img_aug_min_fraction)
    max_fraction = float(args.img_aug_max_fraction)
    current_max = min_fraction + progress * (max_fraction - min_fraction)
    return progress, weight, min_fraction, current_max


def image_aug_lambda(epoch: int, args) -> float:

    return image_aug_schedule(epoch, args)[1]


def image_aug_area_range(epoch: int, args) -> Tuple[float, float]:

    _progress, _weight, min_fraction, current_max = image_aug_schedule(epoch, args)
    return min_fraction, current_max


def _image_aug_rng(seed: int, epoch: int, batch_index: int) -> random.Random:

    mixed = (int(seed) + 700001 * int(epoch) + 1009 * int(batch_index)) % (2**32)
    return random.Random(mixed)


@torch.no_grad()
def make_large_missing_images(
    images: torch.Tensor,
    min_fraction: float,
    max_fraction: float,
    rng: random.Random,
    mean_fill_probability: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if images.ndim != 4:
        raise ValueError("images must be BCHW")
    if not (0.0 < min_fraction <= max_fraction < 0.9):
        raise ValueError("image missing fractions must satisfy 0 < min <= max < 0.9")
    batch, _channels, height, width = images.shape
    mask = images.new_zeros((batch, 1, height, width))
    output = images.clone()
    for index in range(batch):
        fraction = rng.uniform(float(min_fraction), float(max_fraction))
        aspect = rng.uniform(0.55, 1.8)
        block_h = min(
            height - 2,
            max(2, int(round((fraction * height * width / aspect) ** 0.5))),
        )
        block_w = min(width - 2, max(2, int(round(block_h * aspect))))
        y = rng.randint(1, max(1, height - block_h - 1))
        x = rng.randint(1, max(1, width - block_w - 1))
        mask[index, :, y:y + block_h, x:x + block_w] = 1.0

        min_shift_y = max(2, height // 5)
        max_shift_y = max(min_shift_y, height // 2)
        min_shift_x = max(2, width // 5)
        max_shift_x = max(min_shift_x, width // 2)
        shift_y = rng.choice([-1, 1]) * rng.randint(min_shift_y, max_shift_y)
        shift_x = rng.choice([-1, 1]) * rng.randint(min_shift_x, max_shift_x)
        donor = torch.roll(
            images[index:index + 1], shifts=(shift_y, shift_x), dims=(-2, -1)
        )
        if rng.random() < float(mean_fill_probability):
            donor = (
                images[index:index + 1]
                .mean(dim=(-2, -1), keepdim=True)
                .expand_as(donor)
            )
        output[index:index + 1] = (
            images[index:index + 1] * (1.0 - mask[index:index + 1])
            + donor * mask[index:index + 1]
        )
    return output, mask


def image_recovery_cosine_loss(
    target_features: Sequence[torch.Tensor],
    pred_features: Sequence[torch.Tensor],
    image_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    if len(target_features) != len(pred_features):
        raise ValueError("target_features and pred_features must have the same length")
    loss = target_features[0].new_tensor(0.0)
    for target, prediction in zip(target_features, pred_features):
        if target.shape != prediction.shape:
            raise ValueError(
                "Feature shape mismatch: {} vs {}".format(
                    tuple(target.shape), tuple(prediction.shape)
                )
            )
        value = 1.0 - F.cosine_similarity(
            target.detach().float(), prediction.float(), dim=1
        )
        if image_mask is None:
            loss = loss + value.mean()
        else:
            weight = F.interpolate(
                image_mask.float(), size=value.shape[-2:], mode="nearest"
            )[:, 0]
            loss = loss + (value * weight).sum() / weight.sum().clamp_min(1.0)
    return loss


@contextmanager
def freeze_batchnorm_stats(*modules):







    states = []
    for root in modules:
        for module in root.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                states.append(
                    (module, bool(module.training), bool(module.track_running_stats))
                )
                module.train(True)
                module.track_running_stats = False
    try:
        yield
    finally:
        for module, was_training, was_tracking in states:
            module.track_running_stats = was_tracking
            module.train(was_training)





def _masked_spatial_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    v = value.float()
    m = mask.float()
    if v.ndim == 4:
        v = v[:, 0]
    if m.ndim == 4:
        m = m[:, 0]
    num = (v * m).flatten(1).sum(dim=1)
    den = m.flatten(1).sum(dim=1).clamp_min(1.0)
    return (num / den).mean()


def counterfactual_kl_loss(
    q_cf: torch.Tensor,
    clean_target: Optional[torch.Tensor],
    coarse_mask: Optional[torch.Tensor],
    clean_count: int,
) -> torch.Tensor:

    pseudo_count = q_cf.shape[0] - clean_count
    if pseudo_count <= 0:
        return q_cf.new_tensor(0.0)
    if clean_target is None or coarse_mask is None:
        raise ValueError("clean_target and coarse_mask are required for the intervention loss")
    q = q_cf[clean_count:].float().clamp_min(1e-6)
    target = clean_target.float().clamp_min(1e-6)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    kl_map = F.kl_div(q.log(), target, reduction="none").sum(dim=-1)
    return _masked_spatial_mean(kl_map, coarse_mask)


def counterfactual_gain_loss(
    hierarchical_gain: torch.Tensor,
    coarse_mask: Optional[torch.Tensor],
    clean_count: int,
    pseudo_margin: float,
    clean_tolerance: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    b = hierarchical_gain.shape[0]
    pseudo_count = b - clean_count
    zero = hierarchical_gain.new_tensor(0.0)

    clean_loss = zero
    if clean_count > 0:
        clean_excess = F.relu(hierarchical_gain[:clean_count] - float(clean_tolerance))
        clean_loss = clean_excess.mean()

    pseudo_loss = zero
    if pseudo_count > 0:
        if coarse_mask is None:
            raise ValueError("coarse_mask required when pseudo_count > 0")
        shortfall = F.relu(float(pseudo_margin) - hierarchical_gain[clean_count:])
        pseudo_loss = _masked_spatial_mean(shortfall, coarse_mask)

    return clean_loss + pseudo_loss, clean_loss, pseudo_loss


def mixed_original_rd_loss(
    student_input_features: Sequence[torch.Tensor],
    clean_pseudo_targets: Optional[Sequence[torch.Tensor]],
    pred_features: Sequence[torch.Tensor],
    clean_count: int,
    pseudo_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:

    b = pred_features[0].shape[0]
    pseudo_count = b - clean_count

    if clean_count > 0:
        loss_clean = rd_feature_loss(
            [x[:clean_count] for x in student_input_features],
            [x[:clean_count] for x in pred_features],
        )
    else:
        loss_clean = pred_features[0].new_tensor(0.0)

    if pseudo_count > 0:
        if clean_pseudo_targets is None:
            raise ValueError("clean_pseudo_targets required when pseudo_count > 0")
        loss_pseudo = rd_feature_loss(
            list(clean_pseudo_targets),
            [x[clean_count:] for x in pred_features],
        )
    else:
        loss_pseudo = pred_features[0].new_tensor(0.0)

    wc = float(clean_count) / float(max(b, 1))
    wp = (float(pseudo_count) / float(max(b, 1))) * max(0.0, float(pseudo_weight))
    denom = max(wc + wp, 1e-12)
    loss = (wc * loss_clean + wp * loss_pseudo) / denom
    share = wp / denom if pseudo_count > 0 else 0.0
    return loss, loss_clean, loss_pseudo, float(share)


def postfit_state_residual_checkpoint(checkpoint_path: str, train_args):







    from tools import calibrate_state_residual as sclnrm

    post_batch = int(getattr(train_args, "state_residual_batch_size", 128))
    post_workers = int(getattr(train_args, "state_residual_num_workers", -1))
    if post_workers < 0:
        post_workers = int(train_args.num_workers)

    argv = [
        "--data_root", str(train_args.data_root),
        "--checkpoint", str(checkpoint_path),
        "--mode", "fit",
        "--batch_size", str(post_batch),
        "--num_workers", str(post_workers),
        "--seed", str(train_args.seed),
        "--resize", str(train_args.resize),
        "--input_size", str(train_args.input_size),
        "--train_normal_folder", str(train_args.train_normal_folder),
        "--outer_impl", str(train_args.outer_impl),
        "--no_stats_npz",
    ]
    if str(getattr(train_args, "teacher_checkpoint", "")):
        argv.extend(["--teacher_checkpoint", str(train_args.teacher_checkpoint)])
    if str(getattr(train_args, "class_list", "")):
        argv.extend(["--class_list", str(train_args.class_list)])
    if bool(getattr(train_args, "no_recursive_train", False)):
        argv.append("--no_recursive_train")
    if bool(getattr(train_args, "state_residual_export_npz", False)):
        argv.remove("--no_stats_npz")
    if bool(train_args.cpu):
        argv.append("--cpu")
    if bool(train_args.no_deterministic):
        argv.append("--no_deterministic")
    if bool(train_args.allow_tf32):
        argv.append("--allow_tf32")

    cargs = sclnrm.parse_args(argv)
    generator = sclnrm.setup_seed(
        cargs.seed,
        deterministic=not cargs.no_deterministic,
        allow_tf32=cargs.allow_tf32,
    )
    device = torch.device("cpu" if cargs.cpu or not torch.cuda.is_available() else "cuda")

    print("\n" + "=" * 88)
    print("POST-TRAIN prototype-state residual calibration: fitting normal structural-state calibration")
    print("target checkpoint:", checkpoint_path)
    print("external npz export:", "YES" if not cargs.no_stats_npz else "NO (embedded .pth only)")
    print("=" * 88)

    encoder, bottleneck, decoder, ckpt, _ = sclnrm.build_and_load_checkpoint(cargs, device)
    class_names = sclnrm.resolve_class_names(cargs, ckpt)
    train_loader = sclnrm.create_calibration_loader(cargs, class_names, generator)
    
    
    
    
    calibration_dataset = train_loader.dataset

    del train_loader
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_loader = torch.utils.data.DataLoader(
        calibration_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
    )

    print(
        "[prototype-state residual calibration-v3] memory-safe loader: "
        "batch_size=1, num_workers=0, pin_memory=False"
    )
    stats = sclnrm.fit_struct_state_stats(
        cargs, encoder, bottleneck, decoder, train_loader, device, ckpt
    )
    sclnrm.validate_struct_stats(stats)
    sclnrm.embed_struct_stats_in_checkpoint(
        checkpoint_path, stats, calibration_args=cargs, metrics=None
    )
    print("POST-TRAIN prototype-state residual calibration complete; calibration embedded without test evaluation.")
    return stats
