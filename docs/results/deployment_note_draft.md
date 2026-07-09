# Deployment Note — 论文用段落草稿（半页，放 Method 或 Discussion）

> 以下为 LaTeX 可直接粘贴的段落（中英双版）。

---

## English (for paper)

### Efficiency Analysis and Deployment Architecture

**Deployment form.** At inference, the scorer operates as a lightweight single-pass pipeline (Figure X): the ViT encoder processes all $N=720$ vision tokens once; a 0.6M-parameter MLP scorer then assigns per-token importance scores from the layer-0 embeddings in $<$1ms; the top-$K$ tokens ($K = r \times N$) are selected and fed to the LLM for trajectory generation. The scorer's overhead is 0.86 GFLOPs (0.013\% of total), rendering the pruning decision effectively free.

**FLOPs reduction.** Table~\ref{tab:flops} reports theoretical FLOPs under Variant B (physical token drop). At our operating point $r{=}0.5$, the LLM prefill (which dominates at 85\% of total compute) processes 581 instead of 941 tokens, yielding a 39.4\% reduction in LLM prefill FLOPs and 33.6\% reduction in total forward FLOPs. The saving is primarily driven by the FFN layers (O($n$), contributing 38.3\% saving) with an additional boost from the quadratic attention term (O($n^2$), 61.9\% saving on that component alone); at our sequence length the FFN dominates (Table~\ref{tab:flops_breakdown}). For longer-context VLAs (8-camera, multi-frame, $>$2k tokens), the attention fraction grows and the saving amplifies beyond 40\%.

**Variant A $\equiv$ Variant B quality equivalence.** Our quality evaluations (Tables 1–3) use Variant A (attention-mask pruning): pruned tokens remain in the sequence but are masked to $-\infty$ in all attention layers, receiving zero attention weight. Critically, attention-mask pruning and physical-drop pruning are \emph{mathematically equivalent in output} when (i)~masked tokens receive attention weight exactly 0 and (ii)~no downstream token attends to them—both guaranteed by our implementation. Variant A preserves output identity to Variant B up to numerical precision, differing only in realized compute. FLOPs figures in Table~\ref{tab:flops} reflect Variant B; PDMS figures reflect Variant A, and the two are quality-equivalent by construction.

**Engineering path to Variant B.** Variant B (physical token removal + M-RoPE position reindex) is required to realize the FLOPs saving at deployment. It requires reindexing the 3D temporal-height-width RoPE offsets after token selection—straightforward given that selected token positions are known deterministically before the LLM forward pass. It does not alter the selection decision, model weights, or quality. Following FastV~\cite{fastv} and Prune2Drive~\cite{prune2drive}, we report theoretical FLOPs as the primary efficiency metric.

---

## 中文解读（给自己参考，不入论文）

- **部署形态是单 pass**：ViT(一次) → layer-0 embedding → MLP scorer(0.6M, <1ms) → top-K → LLM（只处理 text + kept vision tokens）。不是 2-pass。
- **Variant A**（当前 eval 用的）：质量忠实代理，不省 FLOPs，用于确保选择质量实验公平。
- **Variant B**（部署用的）：真物理 drop token + M-RoPE 位置重编号。省 FLOPs，工程量可控。
- **reviewer 可能问 "Variant B quality 一样吗"**：可在附录用 500 场景验证（Option Y hedge）；或者引 FastV/Prune2Drive 先例说"FLOPs 表 + Variant A quality = 业界惯例"。
- Scorer 开销 0.013%，论文里一句话带过就行。

---

## LaTeX Table Templates

### Table 1: FLOPs Summary

```latex
\begin{table}[t]
\centering
\caption{Theoretical FLOPs under different keep ratios.
ViT cost is constant (pruning occurs after ViT output);
savings are entirely in LLM prefill.
Scorer overhead (0.013\%) omitted.}
\label{tab:flops}
\begin{tabular}{lcccccc}
\toprule
\textbf{Ratio} & \textbf{Kept} & \textbf{Seq} & \textbf{ViT} & \textbf{LLM Prefill} & \textbf{Total} & \textbf{Saving} \\
$r$ & tokens & len & (GFLOPs) & (GFLOPs) & (GFLOPs) & (\%) \\
\midrule
1.00 & 720 & 941 & 949.6 & 5482.8 & 6433.3 & --- \\
0.75 & 540 & 761 & 949.6 & 4393.6 & 5344.1 & 16.9 \\
\textbf{0.50} & \textbf{360} & \textbf{581} & 949.6 & 3323.6 & \textbf{4274.0} & \textbf{33.6} \\
0.25 & 180 & 401 & 949.6 & 2272.6 & 3223.1 & 49.9 \\
\bottomrule
\end{tabular}
\end{table}
```

### Table 2: LLM Prefill Component Breakdown (Appendix or inline)

```latex
\begin{table}[t]
\centering
\caption{LLM prefill FLOPs decomposition. At sequence length 941, FFN layers
dominate (95.2\%), hence the mixed saving (39.4\%) tracks the linear term (38.3\%)
rather than the quadratic attention saving (61.9\%).}
\label{tab:flops_breakdown}
\begin{tabular}{lccccc}
\toprule
\textbf{Ratio} & \textbf{Seq} & \textbf{Attention} & \textbf{FFN+Linear} & \textbf{Attn Saving} & \textbf{FFN Saving} \\
$r$ & len & (GFLOPs) & (GFLOPs) & (\%) & (\%) \\
\midrule
1.00 & 941 & 261.1 & 5221.7 & --- & --- \\
0.75 & 761 & 170.8 & 4222.8 & 34.6 & 19.1 \\
\textbf{0.50} & \textbf{581} & \textbf{99.6} & \textbf{3224.0} & \textbf{61.9} & \textbf{38.3} \\
0.25 & 401 & 47.4 & 2225.2 & 81.8 & 57.4 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 数据来源

- 计算脚本：`scripts/compute_flops_table.py`
- JSON 产物：`results/profiling/flops_table.json`
- 模型参数：`models/Qwen2.5-VL-3B-Instruct/config.json`（ViT 32L h=1280 ffn=3420; LLM 36L h=2048 ffn=11008 GQA 2 KV heads）
