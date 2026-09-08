import argparse
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sprc_rd.data import MultiClassNormalDataset, discover_mvtec_classes, get_data_transforms
from sprc_rd.checkpoint import load_checkpoint_file as _load_checkpoint_file, load_model as _load_integrated_model

EPS = 1e-8


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_seed(seed: int, deterministic: bool=True, allow_tf32: bool=False) -> torch.Generator:
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    try:
        torch.set_float32_matmul_precision('high' if allow_tf32 else 'highest')
    except Exception:
        pass
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def build_and_load_checkpoint(args, device):
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError('checkpoint not found: {}'.format(args.checkpoint))
    ckpt = _load_checkpoint_file(args.checkpoint)
    if not isinstance(ckpt, dict):
        raise TypeError('checkpoint must be a dict, got {}'.format(type(ckpt)))
    missing = [k for k in ('bottleneck', 'decoder') if k not in ckpt]
    if missing:
        raise KeyError('incompatible SPRC-RD checkpoint; missing keys: {}'.format(missing))
    method = str(ckpt.get('method', ''))
    if method and method != 'SPRC-RD':
        raise ValueError('checkpoint method must be SPRC-RD, got {!r}'.format(method))
    integrated, _payload, integrated_cfg = _load_integrated_model(args.checkpoint, device, freeze=True, outer_impl_override='ader')
    print('loaded checkpoint:', args.checkpoint)
    print('checkpoint epoch:', ckpt.get('epoch', 'unknown'))
    print('checkpoint method:', ckpt.get('method', 'unknown'))
    print('backbone:', integrated_cfg.backbone)
    return (integrated.encoder, integrated.bottleneck, integrated.decoder, ckpt, integrated_cfg)


def resolve_class_names(args, ckpt: Optional[Dict]=None) -> List[str]:
    if args.class_list:
        names = [c.strip() for c in args.class_list.split(',') if c.strip()]
        if not names:
            raise ValueError('--class_list was provided but no valid class names were parsed')
        return names
    if ckpt is not None and args.use_checkpoint_classes:
        names = ckpt.get('resolved_class_names', None)
        if isinstance(names, (list, tuple)) and len(names) > 0:
            return [str(x) for x in names]
    if args.class_name.strip().lower() == 'all':
        return discover_mvtec_classes(args.data_root, require_test=True)
    return [args.class_name.strip()]


def create_calibration_loader(args, class_names: Sequence[str], generator: torch.Generator):
    data_transform, _gt_transform = get_data_transforms(args.resize, args.input_size)
    train_data = MultiClassNormalDataset(data_root=args.data_root, class_names=class_names, transform=data_transform, normal_folder=args.train_normal_folder, recursive=not args.no_recursive_train)
    common = {'num_workers': args.num_workers, 'pin_memory': torch.cuda.is_available() and (not args.cpu)}
    if args.num_workers > 0:
        common['worker_init_fn'] = seed_worker
        common['persistent_workers'] = True
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, drop_last=False, generator=generator, **common)
    print('resolved classes ({}): {}'.format(len(class_names), ', '.join(class_names)))
    print('train normal images:', len(train_data))
    if hasattr(train_data, 'class_counts'):
        print('train images per class:', train_data.class_counts())
    return train_loader


@torch.no_grad()
def forward_rcpi(encoder, bottleneck, decoder, img, return_aux: bool=True):
    teacher_features = encoder(img)
    decoder_input, aux = bottleneck(teacher_features, return_aux=return_aux)
    pred_features = decoder(decoder_input)
    if len(teacher_features) != len(pred_features):
        raise RuntimeError('teacher/student layer count mismatch: {} vs {}'.format(len(teacher_features), len(pred_features)))
    return (teacher_features, pred_features, aux)


def compute_native_layer_errors(teacher_features, pred_features) -> List[torch.Tensor]:
    maps = []
    for l, (ft, fp) in enumerate(zip(teacher_features, pred_features)):
        if ft.shape != fp.shape:
            raise RuntimeError('layer {} feature shape mismatch: {} vs {}'.format(l, tuple(ft.shape), tuple(fp.shape)))
        maps.append(1.0 - F.cosine_similarity(ft, fp, dim=1))
    return maps


def _parse_float_list(text: str) -> List[float]:
    vals = []
    for x in str(text).replace(';', ',').split(','):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals


def parse_quantile_probs(spec: str) -> np.ndarray:
    vals = _parse_float_list(spec)
    if not vals:
        vals = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995]
    vals.append(0.5)
    cleaned = []
    for v in vals:
        fv = float(v)
        if not 0.0 < fv < 1.0:
            raise ValueError('quantile probabilities must be in (0,1), got {}'.format(fv))
        cleaned.append(round(fv, 6))
    return np.asarray(sorted(set(cleaned)), dtype=np.float32)


def _quantile_pack(vals: np.ndarray, probs: np.ndarray) -> Dict[str, np.ndarray]:
    vals = np.asarray(vals, dtype=np.float32).reshape(-1)
    if vals.size == 0:
        vals = np.asarray([0.0], dtype=np.float32)
    fixed_probs = np.asarray([0.25, 0.5, 0.75, 0.9, 0.95, 0.99], dtype=np.float32)
    fixed = np.quantile(vals, fixed_probs).astype(np.float32)
    cdf_values = np.quantile(vals, probs).astype(np.float32)
    return {'fixed': fixed, 'cdf_values': cdf_values, 'mean': np.asarray([float(vals.mean())], dtype=np.float32), 'std': np.asarray([max(float(vals.std()), EPS)], dtype=np.float32), 'count': np.asarray([int(vals.size)], dtype=np.int64)}


def _weighted_hist_quantile_pack(hist: np.ndarray, probs: np.ndarray, hist_min: float, hist_max: float) -> Dict[str, np.ndarray]:
    h = np.asarray(hist, dtype=np.float64).reshape(-1)
    B = int(h.size)
    if B < 2:
        raise ValueError('weighted histogram needs at least 2 bins')
    mass = float(h.sum())
    if not np.isfinite(mass) or mass <= EPS:
        zeros = np.zeros(6, dtype=np.float32)
        return {'fixed': zeros, 'cdf_values': np.zeros_like(probs, dtype=np.float32), 'mean': np.asarray([0.0], dtype=np.float32), 'std': np.asarray([1.0], dtype=np.float32), 'weight_mass': np.asarray([0.0], dtype=np.float64)}
    width = float(hist_max - hist_min) / float(B)
    centers = hist_min + (np.arange(B, dtype=np.float64) + 0.5) * width
    csum = np.cumsum(h)

    def wq(q):
        target = float(np.clip(q, 0.0, 1.0)) * mass
        idx = int(np.searchsorted(csum, target, side='left'))
        idx = max(0, min(B - 1, idx))
        prev = 0.0 if idx == 0 else float(csum[idx - 1])
        bin_mass = max(float(h[idx]), EPS)
        frac = float(np.clip((target - prev) / bin_mass, 0.0, 1.0))
        return float(hist_min + (idx + frac) * width)
    fixed_probs = np.asarray([0.25, 0.5, 0.75, 0.9, 0.95, 0.99], dtype=np.float64)
    fixed = np.asarray([wq(q) for q in fixed_probs], dtype=np.float32)
    cdf_values = np.asarray([wq(float(q)) for q in probs], dtype=np.float32)
    mean = float((h * centers).sum() / mass)
    var = float((h * (centers - mean) ** 2).sum() / mass)
    return {'fixed': fixed, 'cdf_values': np.maximum.accumulate(cdf_values).astype(np.float32), 'mean': np.asarray([mean], dtype=np.float32), 'std': np.asarray([max(var ** 0.5, EPS)], dtype=np.float32), 'weight_mass': np.asarray([mass], dtype=np.float64)}


def _entropy_purity(qroute: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qf = qroute.float().clamp_min(EPS)
    entropy = -(qf * qf.log()).sum(dim=-1)
    K = max(int(qroute.shape[-1]), 2)
    purity = (1.0 - entropy / float(np.log(K))).clamp(0.0, 1.0)
    qmax = qroute.max(dim=-1).values
    keff = entropy.exp()
    return (purity.to(dtype=qroute.dtype), qmax, keff)


def _entropy_aware_route_reliability(qroute: torch.Tensor, base_conf: torch.Tensor, args) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    purity, qmax, keff = _entropy_purity(qroute)
    purity_term = purity.pow(float(args.route_entropy_power))
    base_term = base_conf.clamp(0.0, 1.0).pow(float(args.route_base_conf_power))
    joint = torch.sqrt((base_term * purity_term).clamp_min(0.0)).clamp(0.0, 1.0)
    gamma = max(float(args.route_reliability_boost), EPS)
    alpha = 1.0 - (1.0 - joint).clamp(0.0, 1.0).pow(gamma)
    alpha = alpha.clamp(0.0, 1.0)
    return (alpha, {'base_conf': base_conf, 'purity': purity, 'qmax': qmax, 'keff': keff, 'joint': joint})


def _joint_hard_state_from_aux(aux, target_hw, dtype, args):
    argmaxes = []
    alphas = []
    ks = []
    for name in ('fine', 'mid', 'coarse'):
        q = _aligned_cf_q(aux, name, target_hw, dtype)
        ks.append(int(q.shape[-1]))
        conf = _route_confidence_for_layer(aux, name, target_hw, dtype, args.route_confidence_source)
        alpha, _ = _entropy_aware_route_reliability(q, conf, args)
        argmaxes.append(q.argmax(dim=-1))
        alphas.append(alpha)
        del q, conf
    joint_alpha = (alphas[0] * alphas[1] * alphas[2]).clamp_min(0.0).pow(1.0 / 3.0)
    code = (argmaxes[0] * int(ks[1]) + argmaxes[1]) * int(ks[2]) + argmaxes[2]
    return (code, joint_alpha, tuple(ks))


def _decode_state_codes(codes: np.ndarray, ks: Sequence[int]) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    kmul = int(ks[1]) * int(ks[2])
    f = codes // kmul
    rem = codes % kmul
    m = rem // int(ks[2])
    c = rem % int(ks[2])
    return np.stack([f, m, c], axis=1).astype(np.int64)


@torch.no_grad()
def fit_struct_state_stats(args, encoder, bottleneck, decoder, train_loader, device, ckpt):
    print('\n[prototype-state residual calibration-v3] fitting CLASS-AGNOSTIC hierarchical structural-state statistics...')
    print('[prototype-state residual calibration-v3] class indices are ignored for calibration.')
    print('[prototype-state residual calibration-v3] state = (fine cf_q argmax, mid cf_q argmax, coarse cf_q argmax).')
    print('[prototype-state residual calibration-v3] pass 1/2: discovering supported joint states + global residuals...')
    scale_names = ('fine', 'mid', 'coarse')
    hist_bins = int(args.weighted_hist_bins)
    hist_min = float(args.residual_hist_min)
    hist_max = float(args.residual_hist_max)
    hist_scale = float(hist_bins) / float(hist_max - hist_min)
    global_parts = []
    state_mass_all = []
    state_pixel_count_all = []
    ks_per_layer = []
    joint_rel_sum = None
    joint_rel_n = None
    batch_count = 0
    for img, _ignored_class_idx in train_loader:
        img = img.to(device, non_blocking=True)
        teacher_features, pred_features, aux = forward_rcpi(encoder, bottleneck, decoder, img, return_aux=True)
        layer_maps = compute_native_layer_errors(teacher_features, pred_features)
        if not global_parts:
            L = len(layer_maps)
            global_parts = [[] for _ in range(L)]
            joint_rel_sum = np.zeros(L, dtype=np.float64)
            joint_rel_n = np.zeros(L, dtype=np.int64)
            for l, residual in enumerate(layer_maps):
                target_hw = tuple(residual.shape[-2:])
                _code0, _joint0, ks = _joint_hard_state_from_aux(aux, target_hw, residual.dtype, args)
                ks_per_layer.append(ks)
                ncode = int(ks[0] * ks[1] * ks[2])
                del _code0, _joint0
                state_mass_all.append(np.zeros(ncode, dtype=np.float64))
                state_pixel_count_all.append(np.zeros(ncode, dtype=np.int64))
        for l, residual in enumerate(layer_maps):
            target_hw = tuple(residual.shape[-2:])
            code, joint_alpha, ks_now = _joint_hard_state_from_aux(aux, target_hw, residual.dtype, args)
            if tuple(ks_now) != tuple(ks_per_layer[l]):
                raise RuntimeError('prototype counts changed during fitting')
            select = joint_alpha >= float(args.fit_min_joint_reliability)
            weights = joint_alpha * select.to(joint_alpha.dtype)
            r_np = residual.detach().float().cpu().numpy().astype(np.float32)
            global_parts[l].append(r_np.reshape(-1))
            joint_rel_sum[l] += float(weights.sum().detach().cpu())
            joint_rel_n[l] += int(weights.numel())
            cflat = code.reshape(-1)
            wflat = weights.reshape(-1)
            mass = torch.bincount(cflat, weights=wflat.float(), minlength=state_mass_all[l].size)
            cnt = torch.bincount(cflat[select.reshape(-1)], minlength=state_mass_all[l].size) if bool(select.any()) else torch.zeros(state_mass_all[l].size, device=device, dtype=torch.long)
            state_mass_all[l] += mass.detach().cpu().numpy().astype(np.float64)
            state_pixel_count_all[l] += cnt.detach().cpu().numpy().astype(np.int64)
        batch_count += 1
        if args.fit_log_interval > 0 and batch_count % args.fit_log_interval == 0:
            print('  pass1 batches:', batch_count)
    if not global_parts:
        raise RuntimeError('No normal residuals collected. Check data_root/train/good.')
    L = len(global_parts)
    probs = parse_quantile_probs(args.clnrm_cdf_quantiles)
    Q = int(probs.size)
    max_states = int(args.num_structural_states)
    selected_codes = []
    selected_triplets = []
    selected_prior_mass = []
    for l in range(L):
        mass = state_mass_all[l]
        order = np.argsort(-mass)
        keep = order[mass[order] >= float(args.min_state_discovery_mass)]
        if keep.size == 0:
            keep = order[:1]
        keep = keep[:max_states]
        selected_codes.append(keep.astype(np.int64))
        selected_triplets.append(_decode_state_codes(keep, ks_per_layer[l]))
        selected_prior_mass.append(mass[keep].astype(np.float64))
        covered = float(mass[keep].sum() / max(mass.sum(), EPS))
        print('[prototype-state residual calibration-v3] layer {} discovered states: {} / {} possible; weighted coverage={:.4f}'.format(l, len(keep), len(mass), covered))
        topn = min(8, len(keep))
        preview = [(selected_triplets[l][j].tolist(), float(mass[keep[j]])) for j in range(topn)]
        print('  top joint states (fine,mid,coarse; mass):', preview)
    gq25 = np.zeros(L, dtype=np.float32)
    gq50 = np.zeros(L, dtype=np.float32)
    gq75 = np.zeros(L, dtype=np.float32)
    gq90 = np.zeros(L, dtype=np.float32)
    gq95 = np.zeros(L, dtype=np.float32)
    gq99 = np.zeros(L, dtype=np.float32)
    gcdf_values = np.zeros((L, Q), dtype=np.float32)
    gmean = np.zeros(L, dtype=np.float32)
    gstd = np.zeros(L, dtype=np.float32)
    gcount = np.zeros(L, dtype=np.int64)
    for l in range(L):
        vals = np.concatenate(global_parts[l], axis=0).astype(np.float32)
        gp = _quantile_pack(vals, probs)
        gf = gp['fixed']
        gq25[l], gq50[l], gq75[l], gq90[l], gq95[l], gq99[l] = gf
        gcdf_values[l] = gp['cdf_values']
        gmean[l] = float(gp['mean'][0])
        gstd[l] = float(gp['std'][0])
        gcount[l] = int(gp['count'][0])
    del global_parts
    print('[prototype-state residual calibration-v3] pass 2/2: fitting residual histograms for selected joint states...')
    state_hists = [np.zeros((len(selected_codes[l]), hist_bins), dtype=np.float64) for l in range(L)]
    code_maps_cpu = []
    for l in range(L):
        ncode = int(np.prod(ks_per_layer[l]))
        m = np.full(ncode, -1, dtype=np.int64)
        m[selected_codes[l]] = np.arange(len(selected_codes[l]), dtype=np.int64)
        code_maps_cpu.append(m)
    batch_count = 0
    for img, _ignored_class_idx in train_loader:
        img = img.to(device, non_blocking=True)
        teacher_features, pred_features, aux = forward_rcpi(encoder, bottleneck, decoder, img, return_aux=True)
        layer_maps = compute_native_layer_errors(teacher_features, pred_features)
        for l, residual in enumerate(layer_maps):
            target_hw = tuple(residual.shape[-2:])
            code, joint_alpha, ks_now = _joint_hard_state_from_aux(aux, target_hw, residual.dtype, args)
            if tuple(ks_now) != tuple(ks_per_layer[l]):
                raise RuntimeError('prototype counts changed during fitting')
            code_map = torch.as_tensor(code_maps_cpu[l], dtype=torch.long, device=device)
            sid = code_map[code]
            select = (sid >= 0) & (joint_alpha >= float(args.fit_min_joint_reliability))
            if not bool(select.any()):
                continue
            x = residual.clamp(hist_min, np.nextafter(hist_max, hist_min))
            bin_idx = ((x - hist_min) * hist_scale).long().clamp(0, hist_bins - 1)
            s = sid[select]
            b = bin_idx[select]
            w = joint_alpha[select].float()
            flat_index = s * hist_bins + b
            h = torch.bincount(flat_index, weights=w, minlength=len(selected_codes[l]) * hist_bins).reshape(len(selected_codes[l]), hist_bins)
            state_hists[l] += h.detach().cpu().numpy().astype(np.float64)
        batch_count += 1
        if args.fit_log_interval > 0 and batch_count % args.fit_log_interval == 0:
            print('  pass2 batches:', batch_count)
    Smax = max((len(x) for x in selected_codes))
    state_triplets_arr = np.full((L, Smax, 3), -1, dtype=np.int64)
    state_valid_mask = np.zeros((L, Smax), dtype=np.int64)
    state_mass = np.zeros((L, Smax), dtype=np.float64)
    state_shrinkage = np.zeros((L, Smax), dtype=np.float32)
    q25 = np.zeros((L, Smax), dtype=np.float32)
    q50 = np.zeros((L, Smax), dtype=np.float32)
    q75 = np.zeros((L, Smax), dtype=np.float32)
    q90 = np.zeros((L, Smax), dtype=np.float32)
    q95 = np.zeros((L, Smax), dtype=np.float32)
    q99 = np.zeros((L, Smax), dtype=np.float32)
    cdf_values = np.zeros((L, Smax, Q), dtype=np.float32)
    mean = np.zeros((L, Smax), dtype=np.float32)
    std = np.ones((L, Smax), dtype=np.float32)
    tau = max(float(args.state_shrinkage_pixels), 0.0)
    min_mass = float(args.min_state_weight_mass)
    valid_count = 0
    total_selected = 0
    for l in range(L):
        gp_fixed = np.asarray([gq25[l], gq50[l], gq75[l], gq90[l], gq95[l], gq99[l]], dtype=np.float32)
        S = len(selected_codes[l])
        total_selected += S
        state_triplets_arr[l, :S] = selected_triplets[l]
        for s in range(S):
            pp = _weighted_hist_quantile_pack(state_hists[l][s], probs, hist_min=hist_min, hist_max=hist_max)
            mass = float(pp['weight_mass'][0])
            state_mass[l, s] = mass
            if mass >= min_mass:
                alpha = 1.0 if tau <= 0 else mass / float(mass + tau)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                fixed = alpha * pp['fixed'] + (1.0 - alpha) * gp_fixed
                curve = alpha * pp['cdf_values'] + (1.0 - alpha) * gcdf_values[l]
                pm = alpha * float(pp['mean'][0]) + (1.0 - alpha) * float(gmean[l])
                ps = alpha * float(pp['std'][0]) + (1.0 - alpha) * float(gstd[l])
                state_valid_mask[l, s] = 1
                valid_count += 1
            else:
                alpha = 0.0
                fixed = gp_fixed
                curve = gcdf_values[l]
                pm = float(gmean[l])
                ps = float(gstd[l])
            q25[l, s], q50[l, s], q75[l, s], q90[l, s], q95[l, s], q99[l, s] = fixed
            cdf_values[l, s] = np.maximum.accumulate(np.asarray(curve, dtype=np.float32))
            mean[l, s] = float(pm)
            std[l, s] = max(float(ps), EPS)
            state_shrinkage[l, s] = float(alpha)
        for s in range(S, Smax):
            q25[l, s], q50[l, s], q75[l, s], q90[l, s], q95[l, s], q99[l, s] = gp_fixed
            cdf_values[l, s] = gcdf_values[l]
            mean[l, s] = float(gmean[l])
            std[l, s] = float(gstd[l])
    state_span = np.maximum(q99 - q50, EPS).astype(np.float32)
    state_inv_span = (1.0 / state_span).astype(np.float32)
    global_span = np.maximum(gq99 - gq50, EPS).astype(np.float32)
    global_inv_span = (1.0 / global_span).astype(np.float32)
    global_layer_weights = global_inv_span / max(float(global_inv_span.sum()), EPS)
    nstates = np.asarray([len(x) for x in selected_codes], dtype=np.int64)
    proto_counts = np.asarray(ks_per_layer, dtype=np.int64)
    discovery_coverage = np.asarray([float(state_mass_all[l][selected_codes[l]].sum() / max(state_mass_all[l].sum(), EPS)) for l in range(L)], dtype=np.float32)
    stats = {'module': np.asarray(['SPRC-RD-Prototype-State-Residual-Calibration-v3'], dtype='U96'), 'uses_class_condition': np.asarray([0], dtype=np.int64), 'route_source': np.asarray(['joint_relation_guided_cf_q'], dtype='U96'), 'num_layers': np.asarray([L], dtype=np.int64), 'prototype_counts': proto_counts, 'num_states': nstates, 'state_triplets': state_triplets_arr, 'state_valid_mask': state_valid_mask, 'state_weight_mass': state_mass, 'state_shrinkage': state_shrinkage, 'state_q25': q25, 'state_q50': q50, 'state_q75': q75, 'state_q90': q90, 'state_q95': q95, 'state_q99': q99, 'state_cdf_values': cdf_values, 'state_mean': mean, 'state_std': std, 'state_inv_span': state_inv_span, 'global_q25': gq25, 'global_q50': gq50, 'global_q75': gq75, 'global_q90': gq90, 'global_q95': gq95, 'global_q99': gq99, 'global_cdf_values': gcdf_values, 'global_mean': gmean, 'global_std': gstd, 'global_count': gcount, 'global_inv_span': global_inv_span, 'global_layer_weights': global_layer_weights.astype(np.float32), 'cdf_probs': probs, 'discovery_coverage': discovery_coverage, 'mean_fit_joint_reliability': (joint_rel_sum / np.maximum(joint_rel_n, 1)).astype(np.float32), 'hist_min': np.asarray([hist_min], dtype=np.float32), 'hist_max': np.asarray([hist_max], dtype=np.float32), 'hist_bins': np.asarray([hist_bins], dtype=np.int64)}
    if not bool(getattr(args, 'no_stats_npz', False)):
        os.makedirs(os.path.dirname(args.stats_path) or '.', exist_ok=True)
        np.savez(args.stats_path, **stats)
        print('[prototype-state residual calibration-v3] stats saved to:', args.stats_path)
    else:
        print('[prototype-state residual calibration-v3] external npz export disabled; stats kept in memory for checkpoint embedding')
    print('[prototype-state residual calibration-v3] states per layer:', nstates.tolist())
    print('[prototype-state residual calibration-v3] discovery coverage:', discovery_coverage.tolist())
    print('[prototype-state residual calibration-v3] mean fit joint reliability:', stats['mean_fit_joint_reliability'].tolist())
    print('[prototype-state residual calibration-v3] independently supported states: {}/{}'.format(valid_count, total_selected))
    return stats


def _aligned_cf_q(aux, scale_name: str, target_hw: Tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    if scale_name not in aux or 'cf_q' not in aux[scale_name]:
        raise KeyError('SPRC-RD auxiliary output is missing {}.cf_q'.format(scale_name))
    q = aux[scale_name]['cf_q']
    if q.dim() != 4:
        raise ValueError('{}.cf_q must be [B,H,W,K], got {}'.format(scale_name, tuple(q.shape)))
    q = q.to(dtype=dtype)
    if tuple(q.shape[1:3]) != tuple(target_hw):
        q = F.interpolate(q.permute(0, 3, 1, 2).contiguous(), size=target_hw, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous()
    q = q.clamp_min(0.0)
    return q / q.sum(dim=-1, keepdim=True).clamp_min(EPS)


def _route_confidence_for_layer(aux, scale_name: str, target_hw: Tuple[int, int], dtype: torch.dtype, source: str) -> torch.Tensor:
    native = aux[scale_name].get('cf_confidence', None)
    verified = aux.get('prototype_state_confidence', None)
    if native is None or verified is None:
        raise KeyError('SPRC-RD auxiliary output must contain per-scale cf_confidence and prototype_state_confidence')

    def align(x):
        x = x.to(dtype=dtype)
        if tuple(x.shape[-2:]) != tuple(target_hw):
            x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        return x[:, 0].clamp(0.0, 1.0)
    n = align(native)
    v = align(verified)
    source = str(source).lower()
    if source == 'native':
        c = n
    elif source == 'verified':
        c = v
    elif source == 'product':
        c = torch.sqrt((n * v).clamp_min(0.0))
    else:
        raise ValueError('Unsupported route_confidence_source: {}'.format(source))
    return c.clamp(0.0, 1.0)


def _struct_stats_to_checkpoint_payload(stats: Dict[str, np.ndarray]) -> Dict[str, object]:
    out = {}
    for key, value in stats.items():
        if torch.is_tensor(value):
            out[str(key)] = value.detach().cpu().clone()
            continue
        arr = np.asarray(value)
        if arr.dtype.kind in ('U', 'S', 'O'):
            out[str(key)] = arr.tolist()
        else:
            out[str(key)] = torch.from_numpy(np.ascontiguousarray(arr)).clone()
    return out


def embed_struct_stats_in_checkpoint(checkpoint_path: str, stats: Dict[str, np.ndarray], calibration_args=None, metrics=None) -> None:
    validate_struct_stats(stats)
    ckpt = _load_checkpoint_file(checkpoint_path)
    if not isinstance(ckpt, dict):
        raise TypeError('checkpoint must be a dict, got {}'.format(type(ckpt)))
    ckpt['state_residual_stats'] = _struct_stats_to_checkpoint_payload(stats)
    meta = {'format_version': 3, 'module': 'SPRC-RD-Prototype-State-Residual-Calibration-v3', 'embedded': True, 'uses_class_condition': False, 'route_source': 'joint_relation_guided_cf_q'}
    if calibration_args is not None:
        try:
            meta['args'] = dict(vars(calibration_args))
        except TypeError:
            meta['args'] = dict(calibration_args)
    if metrics is not None:
        meta['metrics'] = metrics
    ckpt['state_residual'] = meta
    tmp_path = checkpoint_path + '.state_residual_tmp'
    try:
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, checkpoint_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    print('[prototype-state residual calibration-v3] embedded stats into checkpoint:', checkpoint_path)


def validate_struct_stats(stats: Dict[str, np.ndarray]) -> None:
    required = {'num_layers', 'prototype_counts', 'num_states', 'state_triplets', 'state_shrinkage', 'state_q25', 'state_q50', 'state_q75', 'state_q95', 'state_q99', 'state_cdf_values', 'global_q25', 'global_q50', 'global_q75', 'global_q95', 'global_q99', 'global_cdf_values', 'global_inv_span', 'global_layer_weights', 'state_inv_span', 'cdf_probs'}
    missing = sorted(required.difference(stats.keys()))
    if missing:
        raise KeyError('V3 stats file is missing keys: {}'.format(missing))
    if 'uses_class_condition' in stats and int(np.asarray(stats['uses_class_condition']).reshape(-1)[0]) != 0:
        raise ValueError('Expected class-agnostic V3 stats.')
    if int(np.asarray(stats['num_layers']).reshape(-1)[0]) != 3:
        raise ValueError('V3 expects three RD layers.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SPRC-RD prototype-state residual calibration")
    parser.add_argument("--data_root", default="./mvtec")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outer_impl", default="ader", choices=["ader"])
    parser.add_argument("--teacher_checkpoint", default="")
    parser.add_argument("--stats_path", default="")
    parser.add_argument("--mode", default="fit", choices=["fit"])
    parser.add_argument("--no_stats_npz", action="store_true")
    parser.add_argument("--class_name", default="all")
    parser.add_argument("--class_list", default="")
    parser.add_argument("--use_checkpoint_classes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_normal_folder", default="good")
    parser.add_argument("--no_recursive_train", action="store_true")
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_deterministic", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--fit_log_interval", type=int, default=50)
    parser.add_argument("--num_structural_states", type=int, default=128)
    parser.add_argument("--min_state_discovery_mass", type=float, default=256.0)
    parser.add_argument("--min_state_weight_mass", type=float, default=1024.0)
    parser.add_argument("--state_shrinkage_pixels", type=float, default=8192.0)
    parser.add_argument("--fit_min_joint_reliability", type=float, default=0.05)
    parser.add_argument("--weighted_hist_bins", type=int, default=8192)
    parser.add_argument("--residual_hist_min", type=float, default=0.0)
    parser.add_argument("--residual_hist_max", type=float, default=2.0)
    parser.add_argument("--route_confidence_source", default="product", choices=["native", "verified", "product"])
    parser.add_argument("--route_base_conf_power", type=float, default=1.0)
    parser.add_argument("--route_entropy_power", type=float, default=1.0)
    parser.add_argument("--route_reliability_boost", type=float, default=1.5)
    parser.add_argument("--clnrm_cdf_quantiles", default="0.50,0.60,0.70,0.80,0.85,0.90,0.92,0.94,0.96,0.98,0.99,0.995")
    args = parser.parse_args(argv)
    if not args.stats_path:
        args.stats_path = os.path.splitext(args.checkpoint)[0] + "_state_residual_stats.npz"
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("invalid calibration data-loader settings")
    if args.num_structural_states < 1 or args.weighted_hist_bins < 256:
        raise ValueError("invalid calibration statistics settings")
    return args


def main():
    args = parse_args()
    generator = setup_seed(args.seed, not args.no_deterministic, args.allow_tf32)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    encoder, bottleneck, decoder, checkpoint, _ = build_and_load_checkpoint(args, device)
    class_names = resolve_class_names(args, checkpoint)
    loader = create_calibration_loader(args, class_names, generator)
    stats = fit_struct_state_stats(args, encoder, bottleneck, decoder, loader, device, checkpoint)
    validate_struct_stats(stats)
    embed_struct_stats_in_checkpoint(args.checkpoint, stats, calibration_args=args, metrics=None)


if __name__ == "__main__":
    main()
