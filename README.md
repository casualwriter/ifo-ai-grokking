# Weight Decay as a Necessary Condition for Grokking: Hierarchical Structure Learning in Group Multiplication Tasks

**CK Hung** (Emmy Team, Amoy Studio)    
*Working Paper — June 3, 2026*  
*Zenodo Preprint*  

---

## Abstract

We investigate the role of weight decay in the "grokking" phenomenon, where neural networks achieve perfect training accuracy long before generalizing. Through systematic experiments on the dihedral group $D_{12}$ multiplication task, we report three findings:

1. **Weight decay is necessary, not optional**: With weight decay disabled (WD=0), networks achieve perfect training accuracy but fail to generalize (Val Acc < 15%) even after 25,000 training steps. With WD=0.2, the same networks reach 100% validation accuracy by step 16,000.

2. **Hierarchical structure emerges under compression**: When weight decay is active, networks learn the coarse-grained quotient structure ($\mathbb{Z}_2$ parity, 90% accuracy at step 500) hundreds of steps before element-level prediction (near 0%). This hierarchical progression is absent without weight decay.

3. **Synchronized phase transition at step ~22,000**: All seeds exhibit a brief generalization collapse around step 22,000, driven by weight decay's cumulative compression reaching a critical threshold. This collapse disappears entirely when WD=0, confirming its regularization-induced origin.

We also demonstrate that **functional metrics (Cycle Closure Count) reliably track generalization, while topological metrics (persistent homology $H_1$) do not**. These results frame grokking as a compression-driven phenomenon where regularization forces hierarchical structure extraction.

---

## 1. Introduction

### 1.1 The Grokking Phenomenon

Power et al. (2022) demonstrated that neural networks trained on algorithmic tasks can exhibit "grokking": training accuracy reaches 100% rapidly, but validation accuracy remains near chance for thousands of additional steps before suddenly jumping to near-perfect generalization. This phenomenon challenges conventional intuitions about the relationship between training dynamics and generalization.

Subsequent work has explored various mechanisms:
- **Circuit formation** (Nanda et al., 2023): Sparse algorithmic circuits emerge during the plateau phase
- **Representation compression** (Liu et al., 2023): Weight decay forces efficient encodings
- **Slingshot dynamics** (Thilak et al., 2022): Specific geometric trajectories in parameter space

### 1.2 Open Questions

Despite progress, three questions remain underexplored:

1. **Is weight decay merely accelerating grokking, or is it necessary for grokking to occur at all?**
2. **What internal structure does the network learn during the apparent plateau?**
3. **Do geometric/topological properties of embeddings drive generalization, or are they epiphenomena?**

### 1.3 Our Contributions

Using the dihedral group $D_{12}$ as a controlled testbed, we contribute:

1. **A weight decay ablation study** showing that grokking fails entirely when WD=0, establishing weight decay as a necessary condition rather than an optimization aid.

2. **A novel metric (Cycle Closure Count)** that detects partial structure learning invisible to standard accuracy, revealing that networks learn quotient groups before isomorphisms.

3. **Empirical evidence that persistent homology decouples from generalization**, cautioning against geometric/topological proxies for learning progress.

4. **Identification of a synchronized phase transition at step ~22,000** driven by weight decay reaching critical compression, confirmed by ablation.

---

## 2. Methods

### 2.1 Task: Dihedral Group $D_{12}$

The dihedral group $D_{12}$ has 24 elements representing symmetries of a regular 12-gon:
- 12 rotations: $r^0, r^1, \ldots, r^{11}$
- 12 reflections: $sr^0, sr^1, \ldots, sr^{11}$

Multiplication rules:
$$r^i \cdot r^j = r^{(i+j) \mod 12}, \quad r^i \cdot sr^j = sr^{(j-i) \mod 12}$$
$$sr^i \cdot r^j = sr^{(i+j) \mod 12}, \quad sr^i \cdot sr^j = r^{(j-i) \mod 12}$$

**Element indexing**: We assign indices 0-11 to rotations and 12-23 to reflections (see Section 4.4 for discussion of this choice).

**Task**: Given input pair $(a, b)$, predict $c = a \circ b$.

**Data**: 576 total pairs, split 70/30 (403 train, 173 validation) with fixed seed=42.

### 2.2 Architecture

```
Input (a, b) → Embedding [24×128] → Concat → FC1 [256→128] → ReLU → FC2 [128→24] → Softmax
```

Parameters: ~36k.

### 2.3 Training

- **Optimizer**: AdamW
- **Learning rate**: 1e-3 (constant)
- **Steps**: 25,000-30,000
- **Batch**: Full-batch gradient descent
- **Loss**: Cross-entropy
- **Seeds**: 20 random initializations for main experiments; 2 seeds for detailed ablation

### 2.4 Metrics

**Standard**:
- Training/validation accuracy
- Training loss

**Novel**:

**Cycle Closure Count (CCC)**: For each pair $(a, b)$:
1. Compute $c' = \text{Net}(a, b)$ (network prediction)
2. Compute $c^{-1}$ (true inverse from group table)
3. Test: Does $\text{Net}(c', c^{-1}) = e$ (identity)?

$$\text{CCC} = \frac{|\{(a,b) : \text{Net}(\text{Net}(a,b), c^{-1}) = e\}|}{576}$$

Random baseline (100 untrained networks): CCC = 4.34% ± 6.65%.

**Participation Ratio (PR)**: Effective embedding dimensionality.
$$\text{PR} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$$

**Persistent Homology ($H_1$)**: Maximum 1-D persistence, computed via Ripser with StandardScaler.

**Weight norms**: Frobenius norm of embedding and FC1 matrices.

---

## 3. Results

### 3.1 Standard Grokking Behavior (WD = 0.2)

Across 20 seeds with WD=0.2, all networks exhibited the classic grokking trajectory:

| Phase | Steps | Train Acc | Val Acc | Description |
|-------|-------|-----------|---------|-------------|
| Memorization | 0-4k | 0% → 100% | <50% | Training set fitted |
| Plateau | 4k-12k | 100% | 50-95% | Slow improvement |
| Grokking | 12k-18k | 100% | 95% → 100% | Rapid generalization |
| Post-grok | 18k-22k | 100% | 100% | Stable perfect accuracy |
| Collapse | 22k-23k | ~100% | drops to 80-90% | Phase transition |
| Recovery | 23k-25k | 100% | 90-100% | Reorganization |

We now examine internal dynamics during these phases.

---

### 3.2 Finding 1: Quotient Group Structure Learned First

**Background**: $D_{12}$ has a quotient $D_{12} / \langle r \rangle \cong \mathbb{Z}_2$, partitioning elements into rotations (parity 0) and reflections (parity 1). If networks learn hierarchically, they might master this 2-way distinction before full 24-way classification.

**Experiment**: At step 500 (early memorization), we measured:
- Parity accuracy: Does network's prediction have correct parity?
- CCC conditioned on parity match

**Results across 4 seeds**:

| Seed | Val Acc | CCC | Parity Acc | CCC (Parity Match) | CCC (Parity Mismatch) |
|------|---------|-----|------------|---------------------|------------------------|
| 1 | 0.6% | 53.8% | **90.5%** | 59.5% | 0.0% |
| 2 | 0.0% | 53.3% | **93.2%** | 57.2% | 0.0% |
| 3 | 2.3% | 56.6% | **92.5%** | 61.2% | 0.0% |
| 4 | 0.0% | 53.7% | **83.7%** | 63.7% | 2.1% |

**Random baseline** (100 untrained networks): Parity Acc = 50.1% ± 3.2%.

**Key observations**:

1. **Parity accuracy far exceeds random** (90% vs. 50%) when element accuracy is near 0%.
2. **Closure conditional on parity**: When parity matches, CCC ≈ 60%. When parity mismatches, CCC ≈ 0%. This binary split indicates the network has learned that "rotation × rotation = rotation" and "rotation × reflection = reflection" before learning specific elements.

**Interpretation**: Networks learn the homomorphism $D_{12} \to \mathbb{Z}_2$ before the isomorphism $D_{12} \to D_{12}$. This hierarchical progression is captured by CCC but invisible to standard accuracy.

---

### 3.3 Finding 2: CCC Tracks Generalization, $H_1$ Does Not

**Correlation analysis** across 20 seeds (after step 5,000):

| Metric Pair | Pearson $r$ | Spearman $\rho$ |
|-------------|-------------|-----------------|
| Val Acc ↔ CCC | **0.97** | **0.96** |
| Val Acc ↔ PR | -0.82 | -0.79 |
| Val Acc ↔ $H_1$ | 0.31 | 0.28 |

**Example: $H_1$ decoupling (Seed 100)**:

| Step | Val Acc | CCC | $H_1$ | PR |
|------|---------|-----|-------|-----|
| 15000 | 1.000 | 1.000 | **1.12** | 18.2 |
| 21000 | 1.000 | 1.000 | **0.48** | 15.5 |
| 24000 | 0.960 | 0.988 | 0.52 | 13.7 |

$H_1$ collapsed by 57% while validation accuracy remained perfect.

**Conclusion**: Persistent homology is neither necessary nor sufficient for generalization in this task. CCC provides a more reliable indicator because it tests functional consistency rather than geometric proxies.

---

### 3.4 Finding 3: Synchronized Phase Transition at Step ~22,000

**Observation**: With WD=0.2, all 20 seeds exhibited validation accuracy drops at step ~22,000 (±200 steps), despite different intermediate weight configurations.

**Detailed trajectory (Seed 1, WD=0.2)**:

| Step | Loss | Val Acc | CCC | Emb Norm | FC1 Norm | PR |
|------|------|---------|-----|----------|----------|-----|
| 20000 | 0.00000 | 1.000 | 1.000 | 31.5 | 23.5 | 15.85 |
| 21000 | 0.00000 | 1.000 | 1.000 | 34.0 | 27.8 | 14.81 |
| 22000 | 0.00000 | 0.884 | 0.965 | 44.2 | 42.4 | 13.67 |
| 22200 | 0.00000 | 0.873 | 0.962 | **48.8** | **49.5** | 13.66 |
| 22300 | **0.00002** | 0.832 | 0.786 | 47.7 | 49.1 | 13.70 |
| 23000 | 0.00003 | 0.925 | 0.896 | 41.4 | 42.7 | 13.70 |
| 25000 | 0.00117 | 1.000 | 1.000 | 31.3 | 30.1 | 13.93 |

**Three-phase dynamics**:

1. **Compression (0-21k)**: Weight decay shrinks embedding (56.9 → 31.5), PR drops (19.9 → 14.8)
2. **Critical instability (21k-22.3k)**: Weight norms explode (+40-70%), Val Acc collapses
3. **Reorganization (22.3k-25k)**: Norms restabilize at lower values, generalization recovers

**Cross-seed consistency**: All 20 seeds collapsed within ±200 steps of step 22,000, despite different geometric states at step 21,000 (PR ranging 13.7-21.7).

---

### 3.5 Finding 4: Weight Decay Is Necessary for Grokking (Ablation)

**Experiment**: We reran the entire training protocol with `weight_decay=0.0`, all other parameters identical.

**Results (Seed 1, full trajectory)**:

| Step | Loss | Val Acc | CCC | Emb Norm | FC1 Norm | PR |
|------|------|---------|-----|----------|----------|-----|
| 0 | 3.21 | 0.064 | 0.030 | 56.9 | 6.5 | 19.91 |
| 5000 | 0.00001 | 0.023 | 0.568 | 64.7 | 19.4 | 20.75 |
| 10000 | 0.00000 | 0.052 | 0.573 | 65.9 | 20.8 | 20.85 |
| 15000 | 0.00000 | 0.116 | 0.583 | 66.6 | 21.7 | 20.92 |
| 20000 | 0.00001 | 0.116 | 0.590 | **83.3** | 48.4 | 20.82 |
| 25000 | 0.00000 | **0.145** | 0.625 | 83.3 | 48.4 | 20.82 |

**Results (Seed 2, similar trajectory)**:

| Step | Val Acc | CCC | Emb Norm | PR |
|------|---------|-----|----------|-----|
| 5000 | 0.012 | 0.575 | 63.3 | 20.01 |
| 15000 | 0.046 | 0.585 | 65.2 | 20.25 |
| 25000 | **0.092** | 0.554 | 82.2 | 20.16 |

**Direct comparison at step 25,000**:

| Metric | WD=0.2 | WD=0.0 | Effect |
|--------|--------|--------|--------|
| Val Acc | 1.000 | 0.145 | **-85.5%** |
| CCC | 1.000 | 0.625 | -37.5% |
| Emb Norm | 31.3 | 83.3 | **+166%** |
| FC1 Norm | 30.1 | 48.4 | +61% |
| PR | 13.9 | 20.8 | +50% |

**Three critical findings**:

**(a) No grokking without weight decay**: Val Acc never exceeded 18% across 25,000 steps without WD, compared to 100% by step 16,000 with WD=0.2.

**(b) Weight norms grow monotonically**: Without WD, embedding norms only increase (56.9 → 83.3) and stabilize at high values. With WD, norms compress dramatically (56.9 → 31.3).

**(c) Step-22k collapse vanishes**: Without WD, all metrics evolve smoothly from step 20k onward. The collapse-recovery cycle observed with WD=0.2 is entirely absent.

**Interpretation**: Weight decay is not merely a regularization hyperparameter that improves grokking—it is a *necessary condition* for grokking in this architecture. The compression it enforces drives the hierarchical structure learning observed in Finding 1.

---

## 4. Discussion

### 4.1 Grokking as Compression-Driven Structure Extraction

Our ablation establishes a clear mechanism:

| Condition | Memorization | Compression | Hierarchical Structure | Generalization |
|-----------|--------------|-------------|------------------------|----------------|
| WD = 0.2 | Yes (Step 4k) | Yes (PR: 20 → 14) | Yes (Z₂ at Step 500) | Yes (100%) |
| WD = 0.0 | Yes (Step 4k) | No (PR stays 21) | Partial only | No (15%) |

Without weight decay, networks find a high-dimensional, memorization-based solution. The training loss reaches near-zero, but the representation contains no compressed algebraic structure—only sample-specific patterns.

With weight decay, the network is **forced** to abandon high-dimensional encodings. Under this pressure, only structurally efficient representations survive: first the coarsest possible (Z₂ parity), then progressively finer (full $D_{12}$).

This connects grokking to:
- **Information bottleneck theory** (Tishby & Zaslavsky, 2015): Compression precedes generalization
- **Neural collapse** (Papyan et al., 2020): Late-stage representational simplification
- **Lottery ticket hypothesis** (Frankle & Carbin, 2019): Sparse, efficient subnetworks within dense networks

### 4.2 The Step-22k Phase Transition

We hypothesize the synchronized collapse at step ~22,000 reflects a critical point in the compression dynamics:

**Deterministic origin**: With full-batch training and constant learning rate, weight decay produces near-deterministic exponential compression:
$$\|W_t\|^2 \approx \|W_0\|^2 \cdot (1 - 2\eta\lambda)^t$$

For $\eta = 10^{-3}$, $\lambda = 0.2$, the decay factor per step is $\approx 0.9996$. The cumulative compression reaches a regime around step 22k where:

1. Embedding norms become small enough that softmax logits compress toward uniform
2. Cross-entropy loss gradients spike to maintain training accuracy
3. Weights overshoot (Emb Norm: 34 → 49 in 1,200 steps)
4. Generalization briefly collapses
5. WD reasserts dominance, network reorganizes at higher density

The cross-seed synchronization (±200 steps) reflects the determinism of full-batch dynamics: all networks experience the same compression schedule.

**Confirmation via ablation**: With WD=0, weight norms monotonically increase, never approaching this critical regime. The collapse vanishes.

### 4.3 Hierarchical Learning: Why Quotient First?

Three non-exclusive hypotheses:

**(a) Coding efficiency**: Z₂ classification requires distinguishing 2 classes (1 bit), while full $D_{12}$ requires log₂(24) ≈ 4.6 bits. Under compression pressure, the simplest learnable structure survives first.

**(b) Loss landscape**: The Z₂ classification loss surface may be smoother / more convex than full classification, enabling faster gradient descent.

**(c) Subgroup geometry**: Rotations form a normal subgroup; under appropriate inner product, they may cluster in embedding space, creating natural separation that arises early.

Testing these requires:
- Groups without non-trivial quotients (cyclic primes)
- Groups with multiple quotient structures (symmetric groups)
- Mechanistic interpretability (activation patching for parity circuits)

### 4.4 Limitations

**(a) Single group tested**: $D_{12}$ alone. Generalization to:
- $D_{2n}$ for other $n$
- Symmetric groups $S_n$
- Quaternion group $Q_8$
- Non-trivial direct products

is needed to confirm the necessity of weight decay.

**(b) Element indexing confound**: We assigned indices 0-11 to rotations and 12-23 to reflections (contiguous blocks). The high step-500 parity accuracy could partially reflect this geometric convenience rather than purely learned structure. A controlled experiment with randomized element indices is needed to disentangle these effects.

**(c) Optimizer specificity**: Only AdamW tested. SGD with momentum, vanilla SGD, or other adaptive optimizers may show different dynamics.

**(d) No mechanistic verification**: We document *what* networks learn but not *how*. Identifying parity-computing neurons via activation patching is essential follow-up.

**(e) Architecture**: Only feedforward MLPs tested. Transformer-based grokking may differ.

---

## 5. Future Work

### 5.1 Immediate Extensions (1-2 weeks)

**A. Element indexing ablation**: Re-run experiments with randomly permuted element indices. If quotient-first learning persists, this confirms genuine structural extraction.

**B. Group sweep**: Test $D_8$, $D_{16}$, $S_4$, $S_5$, $Q_8$, $\mathbb{Z}_{24}$, $\mathbb{Z}_{12} \times \mathbb{Z}_2$.

**C. Weight decay schedule**: Test if scheduled WD (high early, low late) accelerates grokking by preventing the step-22k collapse.

### 5.2 Mechanistic Analysis (2-4 weeks)

**D. Parity circuit identification**: Use activation patching to identify which neurons compute Z₂ parity at step 500. Verify causal role.

**E. Phase transition mechanism**: Track Hessian eigenvalues across step 22k to characterize the critical point.

### 5.3 Theoretical Directions (longer-term)

**F. Sample complexity of hierarchical learning**: Prove rigorous bounds on samples required to learn $G/H$ vs. full $G$.

**G. Compression-generalization theorem**: Formalize the relationship between weight decay strength, PR reduction, and generalization gap.

---

## 6. Conclusion

This work establishes three findings about grokking in group multiplication tasks:

1. **Weight decay is necessary for grokking**, not optional. Without it, networks memorize training data in high-dimensional space but fail to generalize.

2. **Hierarchical structure emerges under compression**: Networks first learn coarse quotient groups (Z₂), then refine to full isomorphisms. This progression is invisible to standard accuracy but captured by Cycle Closure Count.

3. **A synchronized phase transition occurs at step ~22,000** when weight decay's cumulative compression reaches a critical point. This transition vanishes when WD=0, confirming its regularization-driven origin.

Additionally, we show that **functional metrics (CCC) outperform topological proxies ($H_1$)** for tracking generalization, cautioning against geometric interpretations of learning dynamics.

These results frame grokking as fundamentally a **compression-driven structure extraction process**, where regularization is the engine rather than a side condition. They suggest that improving grokking dynamics requires understanding and controlling the compression schedule, not just the final compression strength.

---

## Code and Data Availability

All experimental code, trained models, and raw data are available at:
- **Zenodo**: 10.5281/zenodo.20550782
- **GitHub**: https://github.com/casualwriter/ifo-ai-grokking/

Dependencies: Python 3.10, PyTorch 2.0, NumPy 1.24, scikit-learn 1.3, ripser 0.6.

Compute: NVIDIA RTX 4070 / A100 equivalent. ~2 hours for 20-seed run.

---

## Acknowledgments

This work emerged from independent research by CK Hung, with iterative development and peer review by AI assistants (Claude Opus 4, Gemini Flash). 
The author acknowledges the open-source community for tools (Ripser, PyTorch, NumPy) that enabled rapid experimentation.

Funding: None (independent research).
Conflicts of interest: None.

---

## References

Chughtai, B., Chan, L., & Nanda, N. (2023). A toy model of universality: Reverse engineering how networks learn group operations. *arXiv:2302.03025*.

Frankle, J., & Carbin, M. (2019). The lottery ticket hypothesis: Finding sparse, trainable neural networks. *ICLR 2019*.

Liu, Z., Michaud, E. J., & Tegmark, M. (2023). Omnigrok: Grokking beyond algorithmic data. *ICLR 2023*.

Nanda, N., Chan, L., Liberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. *ICLR 2024*.

Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Prevalence of neural collapse during the terminal phase of deep learning training. *PNAS*, 117(40), 24652-24663.

Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. *arXiv:2201.02177*.

Thilak, V., Pillutla, K., Rouditchenko, A., Liu, Z., & Harchaoui, Z. (2022). The slingshot mechanism: An empirical study of adaptive optimizers and the grokking phenomenon. *arXiv:2206.04817*.

Tishby, N., & Zaslavsky, N. (2015). Deep learning and the information bottleneck principle. *IEEE Information Theory Workshop*.

Tralie, C., Saul, N., & Bar-On, R. (2018). Ripser.py: A lean persistent homology library for python. *JOSS*, 3(29), 925.

