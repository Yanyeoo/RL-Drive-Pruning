import torch

# Minimal re-implementation mirroring scripts/train_scorer_budget_rl.py::compute_selection_log_prob
def compute_selection_log_prob(token_scores, B, mode, tau):
    N = token_scores.numel()
    B = max(1, min(B, N))
    eps = 1e-7
    _, top_indices = token_scores.topk(B, dim=0)
    if mode == "softmax":
        logsumexp = torch.logsumexp(token_scores, dim=0)
        log_prob = (token_scores[top_indices].sum() - B * logsumexp) / B
        per_token = token_scores - logsumexp
        return log_prob, top_indices, per_token
    kth = token_scores.topk(B, dim=0).values[-1].detach()
    hard = torch.zeros(N, device=token_scores.device, dtype=token_scores.dtype)
    hard[top_indices] = 1.0
    hard = hard.detach()
    if mode == "st_topk":
        p = torch.sigmoid((token_scores - kth) / tau)
    elif mode == "gumbel":
        g1 = -torch.log(-torch.log(torch.rand_like(token_scores) + eps) + eps)
        g2 = -torch.log(-torch.log(torch.rand_like(token_scores) + eps) + eps)
        p = torch.sigmoid((token_scores - kth + g1 - g2) / tau)
    else:
        raise ValueError(mode)
    per_token = hard * torch.log(p + eps) + (1.0 - hard) * torch.log(1.0 - p + eps)
    log_prob = per_token.mean()
    return log_prob, top_indices, per_token


def main():
    torch.manual_seed(0)
    N, B = 720, 360
    # Simulate a token-score vector with plausible spread (logit-scale)
    s = torch.randn(N, dtype=torch.float32) * 2.0
    s.requires_grad_(True)

    print(f"{'mode':10s} {'log_prob':>12s} {'grad_std':>10s} {'grad_absmax':>12s} "
          f"{'kept_grad_sign':>16s} {'dropped_grad_sign':>18s}")
    for mode, tau in [("softmax", 1.0), ("st_topk", 1.0), ("st_topk", 0.1), ("gumbel", 1.0)]:
        lp, top_idx, per = compute_selection_log_prob(s, B, mode, tau)
        g = torch.autograd.grad(lp, s, retain_graph=True)[0]
        kept_mask = torch.zeros(N, dtype=torch.bool)
        kept_mask[top_idx] = True
        kept_grad = g[kept_mask]
        dropped_grad = g[~kept_mask]
        kept_sign = "+" if kept_grad.mean() > 0 else "-"
        dropped_sign = "+" if dropped_grad.mean() > 0 else "-"
        print(f"{mode:10s} {lp.item():12.5f} {g.std().item():10.5f} {g.abs().max().item():12.5f} "
              f"{kept_sign:>16s} ({kept_grad.mean().item():+.4f})  "
              f"{dropped_sign:>18s} ({dropped_grad.mean().item():+.4f})")

    # Sanity: simulate repeated gradient steps to check scale stays bounded
    print("\n=== scale drift check (50 PG steps, advantage=+1) ===")
    for mode, tau in [("softmax", 1.0), ("st_topk", 1.0), ("st_topk", 0.1), ("gumbel", 1.0)]:
        s2 = torch.randn(N, dtype=torch.float32) * 2.0
        s2.requires_grad_(True)
        adv = 1.0  # positive advantage
        lr = 1e-2
        history = []
        for _ in range(50):
            lp, _, _ = compute_selection_log_prob(s2, B, mode, tau)
            loss = -(adv * lp)
            g = torch.autograd.grad(loss, s2)[0]
            s2 = (s2 - lr * g).detach().requires_grad_(True)
            history.append(s2.detach().std().item())
        print(f"{mode:10s} tau={tau:<4} std: start={history[0]:.3f} end={history[-1]:.3f} "
              f"max={max(history):.3f}  {'<-- BOUNDED' if max(history) < 5.0 else '<-- EXPLODED'}")


if __name__ == "__main__":
    main()
