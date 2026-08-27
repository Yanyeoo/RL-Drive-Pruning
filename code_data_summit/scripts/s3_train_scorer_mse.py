"""s3_train_scorer_mse.py — Train a POINTWISE (MSE) token scorer for:
  (1) C1 ablation: LambdaRank vs MSE (method novelty evidence)
  (2) Calibrated scorer for τ-cut: scores have absolute cross-frame meaning

Key difference from s3_build_labels_train_scorer.py (LambdaRank):
  - Loss: MSE (pointwise regression on L12 attention values), not LambdaRank (pairwise)
  - Label normalization: GLOBAL z-score so scores are cross-frame comparable
  - Same architecture, same data, same split — only the loss changes

Usage:
  cd /apdcephfs/private_shayladeng/tokenrl_autoVLA
  /apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python scripts/s3_train_scorer_mse.py \
    --out-dir ckpt/s3_token_scorer_mse
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
sys.path.insert(0, str(ROOT / "code"))
from rldrive.scoring.token_scorer import TokenImportanceScorer, cam_id_from_blocks, cam_onehot  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feat-dir", default=str(ROOT / "data/s3_scorer/features"))
    p.add_argument("--label-dir", default=str(ROOT / "exp/m1b2_navtrain_full_alllayers"))
    p.add_argument("--out-dir", default=str(ROOT / "ckpt/s3_token_scorer_mse"))
    p.add_argument("--label-layer", type=int, default=12)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-scenes", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    # Label normalization strategy
    p.add_argument("--label-norm", choices=["global", "per_scene", "none"], default="global",
                   help="global: z-score across ALL scenes (cross-frame comparable); "
                        "per_scene: z-score within each scene (like LambdaRank, relative only); "
                        "none: raw attention values")
    return p.parse_args()


def load_dataset(feat_dir, label_dir, label_layer, max_scenes, device):
    feat_dir, label_dir = Path(feat_dir), Path(label_dir)
    toks = sorted(p.stem for p in feat_dir.glob("*.pt"))
    feats, labels = [], []
    vtp0 = blocks0 = None
    n_bad = 0
    for tok in toks:
        lf = label_dir / f"{tok}.pt"
        if not lf.exists():
            continue
        F_data = torch.load(feat_dir / f"{tok}.pt", map_location="cpu", weights_only=False)
        L = torch.load(lf, map_location="cpu", weights_only=False)
        if not torch.equal(F_data["vision_token_positions"], L["vision_token_positions"]):
            n_bad += 1
            continue
        feats.append(F_data["vision_feat"].to(torch.float16))
        labels.append(L["per_layer_vision_attn"][label_layer].mean(0))  # (720,) head-averaged
        if vtp0 is None:
            vtp0 = F_data["vision_token_positions"]; blocks0 = F_data["vision_blocks"]
        if max_scenes and len(feats) >= max_scenes:
            break
    print(f"[train-mse] loaded {len(feats)} scenes ({n_bad} vtp-mismatch dropped)", flush=True)
    feats = torch.stack(feats).to(device)
    labels = torch.stack(labels).to(device).float()
    cam = cam_id_from_blocks(vtp0, blocks0)
    coh = cam_onehot(cam, len(blocks0)).to(device)
    return feats, labels, coh, len(blocks0)


def normalize_labels(labels, split_tr, mode):
    """Normalize attention labels to make scores cross-frame comparable.

    Args:
        labels: (S, 720) raw L12 attention values
        split_tr: indices of training scenes
        mode: 'global' | 'per_scene' | 'none'
    Returns:
        normalized labels, label_mean, label_std (for inverse transform at inference)
    """
    if mode == "none":
        return labels, torch.tensor(0.0), torch.tensor(1.0)
    elif mode == "per_scene":
        # z-score within each scene independently (like LambdaRank, relative only)
        m = labels.mean(dim=1, keepdim=True)
        s = labels.std(dim=1, keepdim=True).clamp(min=1e-6)
        return (labels - m) / s, m.mean(), s.mean()
    elif mode == "global":
        # z-score across ALL training scenes (cross-frame comparable!)
        tr_labels = labels[split_tr]
        label_mean = tr_labels.mean()
        label_std = tr_labels.std().clamp(min=1e-6)
        return (labels - label_mean) / label_std, label_mean, label_std
    else:
        raise ValueError(f"Unknown label-norm mode: {mode}")


def make_inputs(feats_b, coh, mean, std):
    emb = (feats_b.float() - mean) / std
    B = emb.shape[0]
    coh_b = coh.unsqueeze(0).expand(B, -1, -1)
    return torch.cat([emb, coh_b], dim=-1)


@torch.no_grad()
def pairwise_acc(scores, lbl, n_pairs, gen):
    B, N = scores.shape
    i = torch.randint(0, N, (B, n_pairs), generator=gen, device=scores.device)
    j = torch.randint(0, N, (B, n_pairs), generator=gen, device=scores.device)
    si = torch.gather(scores, 1, i); sj = torch.gather(scores, 1, j)
    li = torch.gather(lbl, 1, i); lj = torch.gather(lbl, 1, j)
    sign = torch.sign(li - lj); valid = sign != 0
    correct = ((si - sj) * sign > 0) & valid
    return correct.sum().item() / valid.sum().clamp(min=1).item()


def ndcg_at_k(scores, lbl, k):
    B, N = scores.shape
    gains = lbl.clamp(min=0)
    order = scores.argsort(dim=1, descending=True)[:, :k]
    dcg = (torch.gather(gains, 1, order) / torch.log2(torch.arange(2, k + 2, device=scores.device).float())).sum(1)
    ideal = gains.sort(dim=1, descending=True).values[:, :k]
    idcg = (ideal / torch.log2(torch.arange(2, k + 2, device=scores.device).float())).sum(1)
    return (dcg / idcg.clamp(min=1e-9)).mean().item()


def main():
    a = parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    feats, labels_raw, coh, n_cam = load_dataset(a.feat_dir, a.label_dir, a.label_layer, a.max_scenes, dev)
    S, N, H = feats.shape

    g = torch.Generator(device="cpu").manual_seed(a.seed)
    perm = torch.randperm(S, generator=g)
    n_tr = int(0.8 * S); n_va = int(0.1 * S)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    # Normalize labels (KEY for cross-frame calibration)
    labels, label_mean, label_std = normalize_labels(labels_raw, tr, a.label_norm)
    print(f"[train-mse] label_norm={a.label_norm}, label_mean={label_mean:.6f}, "
          f"label_std={label_std:.6f}", flush=True)

    # Feature norm on train embeddings
    tr_emb = feats[tr].float().reshape(-1, H)
    mean = tr_emb.mean(0); std = tr_emb.std(0).clamp(min=1e-6)

    model = TokenImportanceScorer(H, n_cam).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-5)
    dgen = torch.Generator(device=dev).manual_seed(a.seed)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    logf = (out / "train_log.jsonl").open("w")
    best_va_loss = float("inf"); best_sd = None; patience = 0
    print(f"[train-mse] S={S} N={N} H={H} n_cam={n_cam} train={len(tr)} val={len(va)} "
          f"test={len(te)} loss=MSE label_norm={a.label_norm}", flush=True)

    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        idx = tr[torch.randperm(len(tr), generator=g)]
        tot = 0.0; nb = 0
        for s in range(0, len(idx), a.batch_scenes):
            b = idx[s:s + a.batch_scenes]
            x = make_inputs(feats[b], coh, mean, std)
            scores = model(x)  # (B, 720)
            # MSE loss: pointwise regression on normalized attention values
            loss = F.mse_loss(scores, labels[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            xv = make_inputs(feats[va], coh, mean, std); sv = model(xv)
            va_loss = F.mse_loss(sv, labels[va]).item()
            va_acc = pairwise_acc(sv, labels[va], 4096, dgen)
            va_ndcg = ndcg_at_k(sv, labels[va], N // 2)

        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "val_mse": va_loss,
               "val_pairwise_acc": va_acc, "val_ndcg@360": va_ndcg,
               "lr": sched.get_last_lr()[0], "sec": time.time() - t0}
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        print(f"[train-mse] ep{ep} train_mse={rec['train_loss']:.6f} val_mse={va_loss:.6f} "
              f"val_acc={va_acc:.4f} val_ndcg={va_ndcg:.4f} ({rec['sec']:.1f}s)", flush=True)

        if va_loss < best_va_loss:
            best_va_loss = va_loss
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                print(f"[train-mse] early stop at ep{ep} (best val_mse={best_va_loss:.6f})", flush=True)
                break

    model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        xt = make_inputs(feats[te], coh, mean, std); st = model(xt)
        te_loss = F.mse_loss(st, labels[te]).item()
        te_acc = pairwise_acc(st, labels[te], 8192, dgen)
        te_ndcg = ndcg_at_k(st, labels[te], N // 2)

    # Save
    torch.save(best_sd, out / "checkpoint.pt")
    torch.save({"mean": mean.cpu(), "std": std.cpu()}, out / "feature_norm.pt")
    (out / "config.json").write_text(json.dumps({
        "emb_dim": H, "n_cam": n_cam, "hidden": 256, "label_layer": a.label_layer
    }))
    (out / "manifest.json").write_text(json.dumps({
        "spec": "s3_token_scorer_mse_v1",
        "loss": "MSE (pointwise)",
        "label_norm": a.label_norm,
        "label_mean": float(label_mean),
        "label_std": float(label_std),
        "n_scenes": S, "n_train": len(tr),
        "best_val_mse": best_va_loss,
        "test_mse": te_loss,
        "test_pairwise_acc": te_acc,
        "test_ndcg@360": te_ndcg,
        "seed": a.seed, "epochs_run": ep + 1,
        "features": "layer0_emb+cam_onehot",
        "note": "Pointwise scorer for (1) LambdaRank-vs-MSE ablation and "
                "(2) calibrated τ-cut adaptive pruning. Scores have absolute "
                "cross-frame meaning when label_norm=global."
    }, indent=2))

    # Also save label normalization params (needed for τ-cut at inference)
    torch.save({"label_mean": label_mean.cpu(), "label_std": label_std.cpu()},
               out / "label_norm.pt")

    print(f"[train-mse] DONE. test_mse={te_loss:.6f} test_acc={te_acc:.4f} "
          f"test_ndcg={te_ndcg:.4f} -> {out}", flush=True)
    print(f"[train-mse] For τ-cut: scores have global meaning (label_norm={a.label_norm}). "
          f"Use this scorer with threshold mode to get per-scene adaptive ratios.", flush=True)


if __name__ == "__main__":
    main()
