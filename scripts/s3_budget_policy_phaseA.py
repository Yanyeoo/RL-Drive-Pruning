"""s3_budget_policy_phaseA.py — Phase A: Budget Policy oracle-SFT (claim2).

Near-zero-GPU: reuse the scorer's per-scene PDMS at r in {0.25,0.5,0.75,1.0}
(already computed on navtest_s2sub1500) as oracle; featurize scene_ctx from the
nocot json; train BudgetPolicy(scene_ctx)->4-class; evaluate SIMULATED adaptive
PDMS (exact for a deterministic scorer) vs best-fixed-r vs oracle ceiling.

The PDMS lookup PDMS[token, r_hat] equals what a live per-scene-r run would
produce (scorer selection is deterministic given r), so this is a faithful
claim2 test with no extra GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/apdcephfs/private_shayladeng/tokenrl_autoVLA")
sys.path.insert(0, str(ROOT / "code"))
from rldrive.scoring.budget_policy import BudgetPolicy, build_scene_ctx, RATIOS, CTX_DIM  # noqa: E402


def load_csv_scores(path):
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                d[row["token"]] = float(row["score"])
            except Exception:
                pass
    return d


def score_dist_feats(s: torch.Tensor) -> list:
    """Concentration/redundancy features from the scorer's 720 token scores.
    A peaky distribution => few tokens matter => safe to prune hard."""
    s = s.flatten().float()
    n = s.numel()
    p = torch.softmax(s, 0)
    ent = float(-(p * (p + 1e-12).log()).sum() / torch.log(torch.tensor(float(n))))  # norm entropy
    ss = torch.sort(s, descending=True).values
    psorted = torch.softmax(ss, 0)
    def topmass(k):
        return float(psorted[:max(k, 1)].sum())
    return [ent, topmass(int(0.25*n)), topmass(int(0.5*n)), topmass(int(0.75*n)),
            float(s.std()), float(ss[0] - ss[-1])]


def score_quantiles(s: torch.Tensor, k: int = 48) -> list:
    """k evenly-spaced quantiles of the sorted (desc) score curve = the full
    per-scene importance/redundancy PROFILE (much richer than summary stats)."""
    ss = torch.sort(s.flatten().float(), descending=True).values
    n = ss.numel()
    idx = torch.linspace(0, n - 1, k).round().long()
    q = ss.index_select(0, idx)
    q = (q - q.mean()) / (q.std() + 1e-6)   # per-scene normalize shape (scale-invariant)
    return q.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "ckpt/s3_budget_policy"))
    ap.add_argument("--json-dir", default=str(ROOT / "data/navtest_nocot"))
    ap.add_argument("--feat-dir", default=None, help="if set, augment ctx with scorer score-dist")
    ap.add_argument("--scorer-ckpt", default=str(ROOT / "ckpt/s3_token_scorer"))
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--objective", default="classify", choices=["classify", "regress"],
                    help="classify=CE to oracle r*; regress=predict PDMS per r, argmax at eval")
    ap.add_argument("--score-input", default="dist", choices=["dist", "full"],
                    help="dist=6 summary stats; full=48-quantile score profile (+dist)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    scorer = None
    if a.feat_dir:
        from rldrive.scoring.token_scorer import ScorerRunner
        scorer = ScorerRunner(a.scorer_ckpt, device="cpu")
        print(f"[bp] augmenting ctx with scorer score-dist from {a.feat_dir}", flush=True)

    S3 = ROOT / "results/raw/tokenprune_S3"
    S2 = ROOT / "results/raw/tokenprune_S2"
    pdms = {
        0.25: load_csv_scores(S3 / "S3sub1500_scorer_r025.csv"),
        0.5:  load_csv_scores(S3 / "S3sub1500_scorer_r050.csv"),
        0.75: load_csv_scores(S3 / "S3sub1500_scorer_r075.csv"),
        1.0:  load_csv_scores(S2 / "S2sub1500_attnL12_r100.csv"),
    }
    common = set.intersection(*[set(pdms[r]) for r in RATIOS])
    # featurize
    jdir = Path(a.json_dir)
    toks, X, P = [], [], []
    miss = 0
    for t in sorted(common):
        jp = jdir / f"{t}.json"
        if not jp.exists():
            miss += 1
            continue
        d = json.load(open(jp))
        toks.append(t)
        ctx = build_scene_ctx(d)
        if scorer is not None:
            fp = Path(a.feat_dir) / f"{t}.pt"
            if fp.exists():
                F = torch.load(fp, map_location="cpu", weights_only=False)
                s = scorer.score(F["vision_feat"], F["vision_token_positions"], F["vision_blocks"])
                ctx = ctx + score_dist_feats(s)
                if a.score_input == "full":
                    ctx = ctx + score_quantiles(s, 48)
            else:
                ctx = ctx + [0.0] * (6 + (48 if a.score_input == "full" else 0))
        X.append(ctx)
        P.append([pdms[r][t] for r in RATIOS])
    X = torch.tensor(X, dtype=torch.float32)
    P = torch.tensor(P, dtype=torch.float32)          # (N,4) PDMS at each ratio
    N = len(toks)
    print(f"[bp] N={N} (json miss {miss}) ctx_dim={X.shape[1]} (expect {CTX_DIM})", flush=True)

    # oracle r* label = min r within eps of per-scene max
    mx = P.max(1).values
    lbl = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        for ri, r in enumerate(RATIOS):
            if P[i, ri] >= mx[i] - a.eps:
                lbl[i] = ri
                break
    dist = [int((lbl == k).sum()) for k in range(4)]
    print(f"[bp] oracle r* dist (eps={a.eps}): " +
          " ".join(f"r{RATIOS[k]}={dist[k]}({100*dist[k]/N:.1f}%)" for k in range(4)), flush=True)

    # split
    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(N, generator=g)
    ntr, nva = int(0.7 * N), int(0.15 * N)
    tr, va, te = perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]

    mean = X[tr].mean(0); std = X[tr].std(0).clamp(min=1e-6)
    Xn = (X - mean) / std
    cls_w = torch.tensor([1.0 / max(dist[k], 1) for k in range(4)])
    cls_w = cls_w / cls_w.sum() * 4

    model = BudgetPolicy(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    lossf = torch.nn.CrossEntropyLoss(weight=cls_w)

    def fixed_best(idx):
        m = P[idx].mean(0)
        bi = int(m.argmax()); return RATIOS[bi], float(m[bi])

    def oracle_pdms(idx):
        return float(P[idx].max(1).values.mean())

    def policy_pdms(idx, pred):
        return float(P[idx, pred].mean()), float(torch.tensor([RATIOS[int(c)] for c in pred]).mean())

    best_va = -1; best_sd = None; patience = 0
    for ep in range(a.epochs):
        model.train()
        opt.zero_grad()
        out = model(Xn[tr])
        if a.objective == "regress":
            loss = torch.nn.functional.mse_loss(out, P[tr])
        else:
            loss = lossf(out, lbl[tr])
        loss.backward(); opt.step()
        if ep % 10 == 0 or ep == a.epochs - 1:
            model.eval()
            with torch.no_grad():
                pv = model(Xn[va]).argmax(1)
                va_pdms, _ = policy_pdms(va, pv)
            if va_pdms > best_va:
                best_va = va_pdms
                best_sd = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    break
    model.load_state_dict(best_sd); model.eval()

    with torch.no_grad():
        pte = model(Xn[te]).argmax(1)
    fr, fp = fixed_best(te)
    op = oracle_pdms(te)
    ap_pdms, ap_keep = policy_pdms(te, pte)
    # naive: always the training-majority class
    maj = int(torch.bincount(lbl[tr], minlength=4).argmax())
    naive_pdms = float(P[te, maj].mean())

    print("\n=== Phase A results (TEST split) ===", flush=True)
    print(f" N_test={len(te)}", flush=True)
    print(f" best fixed r={fr}: PDMS={fp:.6f}", flush=True)
    print(f" naive always-r{RATIOS[maj]}: PDMS={naive_pdms:.6f}", flush=True)
    print(f" ADAPTIVE budget policy: PDMS={ap_pdms:.6f} @ mean_keep={ap_keep:.3f}", flush=True)
    print(f" oracle ceiling: PDMS={op:.6f}", flush=True)
    print(f" >>> adaptive - best_fixed = {100*(ap_pdms-fp):+.3f} pt "
          f"(gate: >=+0.5pt for claim2)", flush=True)
    print(f" >>> adaptive captures {100*(ap_pdms-fp)/max(op-fp,1e-9):.1f}% of oracle headroom", flush=True)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(best_sd, out / "checkpoint.pt")
    torch.save({"mean": mean, "std": std}, out / "ctx_norm.pt")
    (out / "config.json").write_text(json.dumps({"ctx_dim": X.shape[1], "hidden": 64, "ratios": RATIOS}))
    (out / "report.json").write_text(json.dumps({
        "N": N, "eps": a.eps, "oracle_r_dist": dist,
        "test": {"n": len(te), "best_fixed_r": fr, "best_fixed_pdms": fp,
                 "naive_pdms": naive_pdms, "adaptive_pdms": ap_pdms, "adaptive_keep": ap_keep,
                 "oracle_pdms": op, "adaptive_minus_fixed_pt": 100 * (ap_pdms - fp)}}, indent=2))
    print(f"[bp] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
