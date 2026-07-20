"""s3_build_labels_train_scorer.py — pair navtrain features with L12-attention
labels and train the LambdaRank token Importance Scorer.

Spec: docs/specs/s3_token_scorer_spec.md.  Run AFTER the feature dump completes
(uses a freed GPU). Pairs by token id; asserts vision_token_positions match.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
sys.path.insert(0, str(ROOT / "code"))
from rldrive.scoring.token_scorer import TokenImportanceScorer, cam_id_from_blocks, cam_onehot  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feat-dir", default=str(ROOT / "data/s3_scorer/features"))
    p.add_argument("--label-dir", default=str(ROOT / "exp/m1b2_navtrain_full_alllayers"))
    p.add_argument("--out-dir", default=str(ROOT / "ckpt/s3_token_scorer"))
    p.add_argument("--label-layer", type=int, default=12)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--pairs-per-scene", type=int, default=1024)
    p.add_argument("--batch-scenes", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
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
        F = torch.load(feat_dir / f"{tok}.pt", map_location="cpu", weights_only=False)
        L = torch.load(lf, map_location="cpu", weights_only=False)
        if not torch.equal(F["vision_token_positions"], L["vision_token_positions"]):
            n_bad += 1
            continue
        feats.append(F["vision_feat"].to(torch.float16))            # (720,H)
        labels.append(L["per_layer_vision_attn"][label_layer].mean(0))  # (720,)
        if vtp0 is None:
            vtp0 = F["vision_token_positions"]; blocks0 = F["vision_blocks"]
        if max_scenes and len(feats) >= max_scenes:
            break
    print(f"[train] loaded {len(feats)} scenes ({n_bad} vtp-mismatch dropped)", flush=True)
    feats = torch.stack(feats).to(device)          # (S,720,H) fp16
    labels = torch.stack(labels).to(device).float()  # (S,720)
    cam = cam_id_from_blocks(vtp0, blocks0)
    coh = cam_onehot(cam, len(blocks0)).to(device)   # (720,n_cam)
    return feats, labels, coh, len(blocks0)


def make_inputs(feats_b, coh, mean, std):
    # feats_b: (B,720,H) fp16 -> standardize -> concat cam onehot (broadcast)
    emb = (feats_b.float() - mean) / std
    B = emb.shape[0]
    coh_b = coh.unsqueeze(0).expand(B, -1, -1)
    return torch.cat([emb, coh_b], dim=-1)          # (B,720,H+n_cam)


def lambdarank_loss(scores, lbl, n_pairs, gen):
    # scores,lbl: (B,720). sample n_pairs (i,j) per scene.
    B, N = scores.shape
    i = torch.randint(0, N, (B, n_pairs), generator=gen, device=scores.device)
    j = torch.randint(0, N, (B, n_pairs), generator=gen, device=scores.device)
    si = torch.gather(scores, 1, i); sj = torch.gather(scores, 1, j)
    li = torch.gather(lbl, 1, i); lj = torch.gather(lbl, 1, j)
    sign = torch.sign(li - lj)
    w = (li - lj).abs()
    valid = sign != 0
    loss = w * torch.nn.functional.softplus(-(si - sj) * sign)
    return (loss * valid).sum() / valid.sum().clamp(min=1)


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
    # mean NDCG@k of scorer ranking vs attention-magnitude gains
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
    feats, labels, coh, n_cam = load_dataset(a.feat_dir, a.label_dir, a.label_layer, a.max_scenes, dev)
    S, N, H = feats.shape

    g = torch.Generator(device="cpu").manual_seed(a.seed)
    perm = torch.randperm(S, generator=g)
    n_tr = int(0.8 * S); n_va = int(0.1 * S)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    # feature norm on train embeddings
    tr_emb = feats[tr].float().reshape(-1, H)
    mean = tr_emb.mean(0); std = tr_emb.std(0).clamp(min=1e-6)

    model = TokenImportanceScorer(H, n_cam).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-5)
    dgen = torch.Generator(device=dev).manual_seed(a.seed)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    logf = (out / "train_log.jsonl").open("w")
    best_va = -1.0; best_sd = None; patience = 0
    print(f"[train] S={S} N={N} H={H} n_cam={n_cam} train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        idx = tr[torch.randperm(len(tr), generator=g)]
        tot = 0.0; nb = 0
        for s in range(0, len(idx), a.batch_scenes):
            b = idx[s:s + a.batch_scenes]
            x = make_inputs(feats[b], coh, mean, std)
            scores = model(x)
            loss = lambdarank_loss(scores, labels[b], a.pairs_per_scene, dgen)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            xv = make_inputs(feats[va], coh, mean, std); sv = model(xv)
            va_acc = pairwise_acc(sv, labels[va], 4096, dgen)
            va_ndcg = ndcg_at_k(sv, labels[va], N // 2)
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "val_pairwise_acc": va_acc,
               "val_ndcg@360": va_ndcg, "lr": sched.get_last_lr()[0], "sec": time.time() - t0}
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        print(f"[train] ep{ep} loss={rec['train_loss']:.4f} val_acc={va_acc:.4f} "
              f"val_ndcg={va_ndcg:.4f} ({rec['sec']:.1f}s)", flush=True)
        if va_acc > best_va:
            best_va = va_acc; best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                print(f"[train] early stop at ep{ep} (best val_acc={best_va:.4f})", flush=True); break

    model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        xt = make_inputs(feats[te], coh, mean, std); st = model(xt)
        te_acc = pairwise_acc(st, labels[te], 8192, dgen)
        te_ndcg = ndcg_at_k(st, labels[te], N // 2)
    torch.save(best_sd, out / "checkpoint.pt")
    torch.save({"mean": mean.cpu(), "std": std.cpu()}, out / "feature_norm.pt")
    (out / "config.json").write_text(json.dumps({"emb_dim": H, "n_cam": n_cam, "hidden": 256,
                                                  "label_layer": a.label_layer}))
    (out / "manifest.json").write_text(json.dumps({
        "spec": "s3_token_scorer_spec_v1", "n_scenes": S, "n_train": len(tr),
        "best_val_pairwise_acc": best_va, "test_pairwise_acc": te_acc, "test_ndcg@360": te_ndcg,
        "seed": a.seed, "epochs_run": ep + 1, "features": "layer0_emb+cam_onehot"}, indent=2))
    print(f"[train] DONE. test_acc={te_acc:.4f} test_ndcg={te_ndcg:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
