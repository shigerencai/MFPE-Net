from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def extract_model_state(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "model_state_dict", "net", "network"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        else:
            state = ckpt
    else:
        raise TypeError("Checkpoint must be a dict or a state_dict.")

    cleaned = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        for prefix in ["module.", "model.", "_orig_mod."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v
    return cleaned


def rename_legacy_mfpe_state_keys(state):
    rename_rules = [
        ("amplinet.", "intensity_evolution_branch."),
        ("phasenet.", "position_evolution_branch."),
        ("phy_branch.", "physical_evolution_branch."),
        ("alphamixer.", "confidence_guided_fusion_module."),
        ("high_refiner.", "strong_echo_lead_time_residual_refinement_module."),
        (".phase_embed.", ".phase_feature_encoder."),
        (".spatial_embed.", ".spatial_motion_encoder."),
        (".offset_net.", ".phase_offset_estimator."),
        (".intensity_dec.", ".intensity_residual_head."),
        (".local_motion_dec.", ".local_motion_residual_head."),
        (".evo_net.", ".physical_evolution_estimator."),
        (".gamma", ".intensity_residual_gate"),
        (".confidence.", ".confidence_estimator."),
        (".residual.", ".residual_fusion_branch."),
    ]
    renamed = {}
    for key, value in state.items():
        new_key = key
        for old, new in rename_rules:
            new_key = new_key.replace(old, new)
        renamed[new_key] = value
    return renamed


def _filter_compatible_state(model, state, strict = False):
    if strict:
        return state, []
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            filtered[k] = v
        else:
            skipped.append(k)
    return filtered, skipped


def load_model_weights(model, ckpt_path, device, strict = False):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = rename_legacy_mfpe_state_keys(extract_model_state(ckpt))
    state, skipped = _filter_compatible_state(model, state, strict=strict)
    result = model.load_state_dict(state, strict=strict)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys) + [f"skipped_incompatible::{k}" for k in skipped]
    return missing, unexpected


def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step,
                    best_score, args_dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "args": args_dict,
    }, path)


def resume_checkpoint(path, model, optimizer, scheduler, device, load_optimizer = True,
                      strict = False):
    ckpt = torch.load(path, map_location=device)
    state = rename_legacy_mfpe_state_keys(extract_model_state(ckpt))
    state, skipped = _filter_compatible_state(model, state, strict=strict)
    result = model.load_state_dict(state, strict=strict)

    if load_optimizer and optimizer is not None and ckpt.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[WARN] Optimizer state was not loaded: {e}")
    if load_optimizer and scheduler is not None and ckpt.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print(f"[WARN] Scheduler state was not loaded: {e}")

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    global_step = int(ckpt.get("global_step", 0))
    best_score = float(ckpt.get("best_score", -1.0))
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys) + [f"skipped_incompatible::{k}" for k in skipped]
    return start_epoch, global_step, best_score, missing, unexpected
