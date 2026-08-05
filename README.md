# Master's Thesis — Progress Summary

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

> Gemma α=5 was forced by catastrophic generation at higher values (activation magnitudes at k\*=39 are much larger than other models).

### Languages
- **13 HRL:** English, German, Japanese, Korean, Chinese, Russian, Thai, Arabic, Spanish, French, Italian, Dutch, Polish
- **3 LRL:** Yoruba (yo), Swahili (sw), Amharic (am)

### Method
- **Jailbreak vector j_l**: DiM (Difference-in-Means) between bypassed and refused activation states
- **Refusal direction r_l**: DiM between harmful and harmless activation states
- **k\***: argmin layer-wise cos(ĵ_l, r̂_l) — layer of maximum anti-parallel alignment
- **Attack (RQ2)**: h'[k\*] = h[k\*] + α · ĵ_src → measure bypass rate increase
- **Defense (RQ3)**: h'[k\*] = h[k\*] − α · ĵ_src → measure conditional recovery rate
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
| Swahili (LRL)  | n/a\*     | −0.39        | −0.31      |
| Amharic (LRL)  | n/a\*     | **+0.52**    | −0.36      |

\* sw/am on Qwen: bypass ≈98%, too few refused samples to extract j_l

**Finding:** HRL jailbreak vectors are strongly anti-parallel to refusal directions across all models. LRL languages show near-orthogonal or anomalous structure — notably Amharic on Llama (+0.52), the largest positive value observed.

---

### RQ2 — Cross-lingual Attack Transfer (complete, all 3 models)

#### Qwen2.5-7B (k\*=18, α=20)

| Target | Baseline | en→ | zh→ | ko→ | yo→ |
|--------|----------|-----|-----|-----|-----|
| en (HRL) | 9.8% | **75.7%** | **65.2%** | **70.8%** | 11.4% |
| de (HRL) | 15.0% | **81.3%** | **72.7%** | **78.8%** | 14.0% |
| zh (HRL) | 16.1% | **77.4%** | **72.6%** | **81.6%** | 18.2% |
| ko (HRL) | 26.0% | **67.0%** | **64.2%** | **68.2%** | 26.7% |
| ja (HRL) | 24.7% | **68.9%** | **69.2%** | **74.0%** | 29.0% |
| yo/sw/am (LRL) | 87–98% | ≈99% | ≈99% | ≈99% | ≈98% *(ceiling)* |

→ HRL sources +41–67pp; Yoruba source ≈0pp (geometric null, matches RQ1).

#### Llama-3.1-8B (k\*=25, α=20) — LRL source anomaly

| Target | Baseline | zh→ | ko→ | yo→ | sw→ | am→ |
|--------|----------|-----|-----|-----|-----|-----|
| en (HRL) | 1.9% | 66.1% | 78.3% | 61.4% | 76.0% | **90.0%** |
| de (HRL) | 3.3% | 79.0% | 85.1% | 67.7% | 77.6% | **95.1%** |
| zh (HRL) | 13.3% | 77.8% | **93.9%** | 71.0% | 84.4% | 85.0% |
| ko (HRL) | 33.9% | 76.7% | 89.9% | 73.8% | 57.7% | **96.9%** |
| ja (HRL) | 23.6% | 63.3% | 87.2% | 78.1% | **91.6%** | 92.5% |
| yo/sw/am (LRL) | 79–87% | ≈89% | ≈78% | ≈88% | ≈91% | ≈91% *(ceiling)* |

→ **All sources — including LRL — produce strong HRL transfer (+56–93pp).** Amharic (cos=+0.52 in RQ1) is the strongest attacker. This breaks the geometric prediction from Qwen.

#### Gemma-2-9B (k\*=39, α=5) — null result

All bypass rates unchanged (±0.5pp from baseline) across every source-target pair. α=5 is too small to perturb this model's activation space. Results are uninformative about transfer capacity.

---

### RQ3 — Cross-lingual Defense (complete, all 3 models)

Metric: **Conditional Recovery Rate** = fraction of originally-bypassed samples that refuse after defense. Input: bypassed samples only (baseline compliance = 100%).

#### Qwen2.5-7B (k\*=18, α=20)

| Target | Type | en− | zh− | ko− | ar− | yo− |
|--------|------|-----|-----|-----|-----|-----|
| en | HRL | **96.4%** | 96.4% | 98.2% | 98.2% | 10.7% |
| de | HRL | **90.7%** | 89.5% | 89.5% | 89.5% | 12.8% |
| zh | HRL | 87.0% | *80.4%* | 83.7% | 78.3% | 10.9% |
| ko | HRL | 73.8% | 73.2% | *73.2%* | **78.5%** | 18.1% |
| ja | HRL | 81.6% | 78.7% | 80.9% | **85.1%** | 16.3% |
| ru | HRL | **93.2%** | 88.1% | 94.9% | 94.9% | 8.5% |
| ar | HRL | 92.5% | 85.8% | 89.2% | *91.7%* | 10.8% |
| yo | LRL | 62.6% | 58.8% | 73.2% | **74.8%** | *57.0%* |
| sw | LRL | 27.5% | 31.5% | 28.3% | **32.6%** | 15.2% |
| am | LRL | **30.8%** | 23.8% | 24.2% | 24.0% | 29.5% |

*Italics = self-defense*

→ HRL targets: 73–98% recovery. yo target: 63–75% (cross-lingual HRL beats self-defense 57%). sw/am: 24–33% (fundamental limit).

#### Llama-3.1-8B (k\*=25, α=20) — am anomaly

| Target | Type | zh− | ru− | ar− | th− | yo− | am− |
|--------|------|-----|-----|-----|-----|-----|-----|
| en | HRL | 100.0% | 100.0% | 100.0% | 90.9% | 100.0% | 72.7% |
| de | HRL | 84.2% | 100.0% | 89.5% | 100.0% | 84.2% | 42.1% |
| zh | HRL | *85.5%* | 61.8% | 89.5% | 81.6% | 76.3% | 82.9% |
| ja | HRL | 49.6% | 64.4% | 37.8% | 60.0% | 63.0% | 25.2% |
| ko | HRL | 74.7% | 75.8% | 72.7% | 54.6% | 69.1% | 21.1% |
| th | HRL | 77.6% | 82.8% | 72.4% | *94.8%* | 74.1% | **15.5%** |
| ar | HRL | 81.6% | 65.8% | *73.7%* | 78.9% | 84.2% | 39.5% |
| yo | LRL | 58.0% | 55.0% | **79.5%** | 50.1% | *53.3%* | 14.8% |
| sw | LRL | 40.4% | 48.4% | 36.9% | 46.6% | 49.0% | **7.9%** |
| am | LRL | 53.3% | **72.9%** | 33.7% | 44.5% | 55.9% | *24.6%* |

*Italics = self-defense*

→ zh/ru/th give 50–100% HRL recovery. **am source: weakest defense (8–83%, mean ≈44%) despite being the strongest attacker in RQ2.** This is the clearest empirical dissociation between attack transferability and defensive intervenability.

#### Gemma-2-9B (k\*=39, α=5) — near-zero

Recovery rates: 0–25% across all source-target pairs. Most HRL targets: 0–12%. LRL targets: 2–10%. Best cases: ja→23% (ko source), en/nl→25% (selected sources). α=5 constraint dominates; results inconclusive.

---

## Key Theoretical Contributions

### 1. Three-Concept Dissociation

Three properties are empirically distinguishable and have different requirements:

| Property | Measure | HRL (Qwen) | Yoruba (Qwen) | am (Llama) |
|----------|---------|-----------|----------------|------------|
| Representation sharing | cos(ĵ_src, r̂_tgt) | Strong (−0.5 to −0.7) | Absent (+0.22) | Reversed (+0.52) |
| Attack transferability | Δ bypass rate | +41–67pp | ≈0pp | **+56–93pp** |
| Defensive intervenability | Recovery rate | 73–98% | 57% (self) / 75% (cross) | **8–83%** |

The Amharic-on-Llama case is the clearest dissociation: strongest attacker, weakest defender, anomalous geometry.

### 2. Attack–Defense Asymmetry

Attack transfer is tolerant of geometric mismatch (Llama LRL sources all transfer effectively); defense is not. The geometric requirements for suppressing refusal (+j) differ from those for inducing refusal (−j).

### 3. Geometric Conditionality

Cross-lingual transferability is not universal. It is conditioned on cos(ĵ_src, r̂_tgt) — but this geometric predictor is model-dependent: it holds precisely for Qwen, is violated by Llama, and is masked by α constraints in Gemma.

---

## Experiments in Progress

| Experiment | Status | Purpose |
|------------|--------|---------|
| Over-refusal measurement | Running on cluster | Verify defense doesn't over-refuse harmless inputs |
| Random vector control | Script ready, not yet submitted | Confirm j_src *direction* (not magnitude) drives effects |

### Random Vector Control Logic
Replace ĵ_src with a random unit vector (same dimension, same α, same k\*, n=5 seeds). Expected result: random vectors produce near-baseline bypass (attack) and near-zero recovery (defense) → confirms j_src direction encodes safety-relevant information.

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
└── thesis/                            # LaTeX thesis (chapters/results.tex updated)
```
