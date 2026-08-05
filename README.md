# Master's Thesis — Progress Summary

**Title:** On the Limits of Cross-Lingual Inference-Time Safety Steering: Geometry, Transfer, and Defense in Multilingual Language Models

**Author:** Baoying Gao (baga0553) — Stockholm University, DSV

---

## Research Overview

This thesis investigates whether the internal geometric structure that underlies refusal behavior in high-resource languages (HRL) can be exploited across linguistic boundaries — both as an attack vector and as a training-free defense mechanism. It extends Wang et al. (2026) *"Refusal Direction is Universal Across Safety-Aligned Languages"* by evaluating **when** cross-lingual transfer succeeds and **when** it fails.

The core claim is that **representation sharing, attack transferability, and defensive intervenability are three empirically dissociable properties** — not a single unified phenomenon.

---

## Research Questions

| RQ | Question |
|----|----------|
| **RQ1** | Do jailbreak vectors and refusal directions share geometric structure across languages, and does this differ between HRL and LRL languages? |
| **RQ2** | Does geometric compatibility between source and target safety representations predict cross-lingual attack transferability? |
| **RQ3** | Are shared cross-lingual safety representations sufficient to support inference-time defense, and under what conditions does this approach fail? |

---

## Experimental Setup

### Models
| Model | Parameters | k* | α |
|-------|-----------|-----|---|
| Qwen2.5-7B-Instruct | 7B | 18 | 20.0 |
| Llama-3.1-8B-Instruct | 8B | 25 | 20.0 |
| Gemma-2-9B-IT | 9B | 39 | 5.0 |

### Languages
- **13 HRL:** English, German, Japanese, Korean, Chinese, Russian, Thai, Arabic, Spanish, French, Italian, Dutch, Polish
- **3 LRL:** Yoruba (yo), Swahili (sw), Amharic (am)

### Method
- **Jailbreak vector j_l**: DiM (Difference-in-Means) between bypassed and refused activation states
- **Refusal direction r_l**: DiM between harmful and harmless activation states
- **Key geometry**: cos(ĵ_l, r̂_l) measured layer-wise; intervention layer k* = argmin cos
- **Attack (RQ2)**: h'[k*] = h[k*] + α · ĵ_src  →  measure bypass rate increase
- **Defense (RQ3)**: h'[k*] = h[k*] − α · ĵ_src  →  measure conditional recovery rate
- **Evaluation**: WildGuard classifier (binary refusal label)
- **Dataset**: PolyRefuse benchmark (harmful + harmless, 16 languages)

---

## Results

### RQ1 — Geometric Structure (complete, all 3 models)

Average layer-wise cosine similarity cos(ĵ_l, r̂_l):

| Language group | Qwen2.5-7B | Llama-3.1-8B | Gemma-2-9B |
|----------------|-----------|--------------|------------|
| HRL mean       | −0.47     | −0.67        | −0.67      |
| Yoruba (LRL)   | +0.22     | −0.05        | −0.02      |
| Swahili (LRL)  | n/a*      | −0.39        | −0.31      |
| Amharic (LRL)  | n/a*      | **+0.52**    | −0.36      |

\* sw/am on Qwen: bypass ≈98%, too few refused samples to extract j_l

**Finding:** HRL jailbreak vectors are strongly anti-parallel to refusal directions (cos ≈ −0.5 to −0.7), consistent across all models. LRL languages show near-orthogonal or anomalous structure — particularly Amharic on Llama (+0.52), the strongest positive alignment observed.

---

### RQ2 — Cross-lingual Attack Transfer (Qwen complete; Llama/Gemma pending)

Bypass rate after injecting ĵ_src at k*=18 (Qwen2.5-7B, α=20):

| Target | Baseline | en→ | zh→ | ko→ | yo→ |
|--------|----------|-----|-----|-----|-----|
| en (HRL) | 9.8%  | **75.7%** | **65.2%** | **70.8%** | 11.4% |
| de (HRL) | 15.0% | **81.3%** | **72.7%** | **78.8%** | 14.0% |
| zh (HRL) | 16.1% | **77.4%** | **72.6%** | **81.6%** | 18.2% |
| ko (HRL) | 26.0% | **67.0%** | **64.2%** | **68.2%** | 26.7% |
| ja (HRL) | 24.7% | **68.9%** | **69.2%** | **74.0%** | 29.0% |
| yo (LRL) | 87.4% | 98.3% | 98.3% | 97.9% | 98.1% *(ceiling)* |
| sw (LRL) | 97.7% | 99.7% | — | — | — *(ceiling)* |
| am (LRL) | 98.3% | 99.5% | — | — | — *(ceiling)* |

**Finding:** HRL source vectors raise HRL bypass rates by +41–67 pp. Yoruba vector produces near-zero transfer (+1–4 pp) — directly predicted by its near-orthogonal geometry. LRL targets show ceiling effects (baseline already 87–98%).

---

### RQ3 — Cross-lingual Defense (Qwen complete; Llama/Gemma pending)

Conditional recovery rate after subtracting ĵ_src at k*=18 (Qwen2.5-7B, α=20). Metric = fraction of originally bypassed samples that refuse after intervention. Input: bypassed samples only (baseline compliance = 100%).

| Target | Type | en− | zh− | ko− | ar− | yo− | self |
|--------|------|-----|-----|-----|-----|-----|------|
| en | HRL | **96.4%** | 96.4% | 98.2% | 98.2% | 10.7% | *96.4%* |
| de | HRL | **90.7%** | 89.5% | 89.5% | 89.5% | 12.8% | *90.7%* |
| zh | HRL | 87.0% | *80.4%* | 83.7% | 78.3% | 10.9% | *80.4%* |
| ko | HRL | 73.8% | 73.2% | *73.2%* | 78.5% | 18.1% | *73.2%* |
| ja | HRL | 81.6% | 78.7% | 80.9% | **85.1%** | 16.3% | *82.3%* |
| ru | HRL | 93.2% | 88.1% | **94.9%** | 94.9% | 8.5% | *88.1%* |
| ar | HRL | 92.5% | 85.8% | 89.2% | *91.7%* | 10.8% | *91.7%* |
| yo | LRL | 62.6% | 58.8% | **73.2%** | **74.8%** | *57.0%* | *57.0%* |
| sw | LRL | 27.5% | 31.5% | 28.3% | **32.6%** | 15.2% | n/a |
| am | LRL | **30.8%** | 23.8% | 24.2% | 24.0% | 29.5% | n/a |

*Italics = self-defense (same-language source); bold = best per row*

**Finding:**
- **HRL targets**: 73–98% recovery — high and robust across all HRL source vectors
- **Yoruba**: 63–75% recovery; cross-lingual HRL defenders *outperform* self-defense (57%) — consistent with yo's near-orthogonal j_l being a poor suppressor
- **Swahili/Amharic**: only 24–33% recovery despite HRL source vectors — fundamental limit, not hyperparameter failure
- **Yoruba source on any target**: 9–18% recovery — matches geometric prediction (orthogonal vector provides no refusal signal)

---

## Key Theoretical Contributions

### 1. Three-Concept Dissociation
Three properties are empirically distinguishable:

| Property | Measure | HRL | Yoruba (Qwen) | sw/am (Qwen) |
|----------|---------|-----|----------------|--------------|
| Representation sharing | cos(ĵ_src, r̂_tgt) | Strong (−0.5 to −0.7) | Absent (+0.22) | Weak (−0.3 to −0.4) |
| Attack transferability | Δ bypass rate | +41–67 pp | ≈0 pp | ceiling effect |
| Defensive intervenability | Recovery rate | 73–98% | 57% (self) / 75% (cross) | 24–33% |

Representation sharing → attack transfer follows geometrically. But defense is more brittle: shared representations are *necessary but not sufficient* for defense to work.

### 2. Attack–Defense Asymmetry
Attack transfer is relatively tolerant of geometric mismatch; defense is not. When the target language lacks coherent refusal geometry, attack still works via ceiling effects, but defense cannot reinforce a mechanism that does not exist.

### 3. Geometric Conditionality
Cross-lingual transferability is not universal. It is conditioned on cos(ĵ_src, r̂_tgt): where this cosine is strongly negative, both attack and defense transfer; where it is near zero or positive, both fail.

---

## Experiments in Progress / Planned

| Experiment | Status | Purpose |
|------------|--------|---------|
| RQ2/RQ3 for Llama-3.1-8B | Results available, writing pending | Replication + anomaly (am cos=+0.52 yet strong attack) |
| RQ2/RQ3 for Gemma-2-9B | Results available, writing pending | Replication (α=5, constrained) |
| Over-refusal measurement | Running on cluster (~1.3h/model) | Verify defense doesn't refuse harmless inputs |
| Random vector control | Script ready, not yet submitted | Confirm that j_src *direction* (not magnitude) drives effects |

### Random Vector Control Logic
Replace ĵ_src with a random unit vector of the same dimension, same α, same k*, n=5 seeds. Expected: random vectors produce near-baseline bypass rates (attack) and near-zero recovery (defense) → confirms j_src direction is geometrically meaningful.

---

## Repository Structure

```
experiment_thesis/
├── scripts/
│   ├── extract_jailbreak_vectors.py   # RQ1: DiM extraction, geometry analysis
│   ├── run_baseline.py                # Baseline bypass/refusal rates
│   ├── cross_lingual_transfer.py      # RQ2: attack transfer
│   ├── cross_lingual_defense.py       # RQ3: defense intervention
│   ├── over_refusal.py                # Over-refusal on harmless inputs
│   └── random_vector_control.py       # Direction specificity control
├── slurm/                             # SLURM job scripts for each experiment
├── output/
│   ├── jailbreak_analysis/            # j_l vectors + geometry_results.json
│   ├── defense/                       # Defense results per model
│   └── transfer/                      # Transfer results per model
└── thesis/                            # LaTeX thesis
```
