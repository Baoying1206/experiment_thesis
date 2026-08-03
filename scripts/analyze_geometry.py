"""
Generate all thesis figures from geometry_results.json files.

Figures produced:
  Figure 1 (RQ1): Heatmap of avg cosine similarity — jailbreak lang × refusal lang, per model
  Figure 2 (RQ1): Layer-wise cosine similarity profiles for same-language pairs
  Figure 3 (RQ2-geo): Cross-lingual jailbreak vector similarity heatmap, per model
  Figure 4:       Three-model comparison — diagonal cosine similarity by language
  Figure 5:       Yoruba anomaly highlight
  Figure 6:       L2 norm profiles per layer
  Figure 7 (RQ1): Three-way geometric relationship
  Figure 8 (RQ2-beh): Cross-lingual transfer bypass rate matrix

Usage:
  python scripts/analyze_geometry.py \
      --results_dir  output/jailbreak_analysis \
      --transfer_dir output/transfer \
      --output_dir   output/figures
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Language ordering ─────────────────────────────────────────────────────────
ALL_LANGS = ['en', 'de', 'ja', 'ko', 'zh', 'ru', 'th', 'yo', 'ar', 'es', 'fr', 'it', 'nl', 'pl', 'sw', 'am']

# Language resource level classification
LRL_LANGS = {'yo', 'sw', 'am'}   # low-resource: no/minimal safety alignment
HRL_LANGS = {'en', 'de', 'es', 'fr', 'it', 'nl', 'pl', 'zh', 'ru', 'ja', 'ko', 'ar', 'th'}

RESOURCE_COLORS = {'HRL': '#2196F3', 'LRL': '#E53935', None: 'gray'}

MODELS = [
    'Qwen2.5-7B-Instruct',
    'Meta-Llama-3.1-8B-Instruct',
    'gemma-2-9b-it',
]

MODEL_LABELS = {
    'Qwen2.5-7B-Instruct':          'Qwen2.5-7B',
    'Meta-Llama-3.1-8B-Instruct':   'LLaMA-3.1-8B',
    'gemma-2-9b-it':                'Gemma-2-9B',
}

COLORS = {
    'Qwen2.5-7B-Instruct':          '#2196F3',
    'Meta-Llama-3.1-8B-Instruct':   '#FF5722',
    'gemma-2-9b-it':                '#4CAF50',
}



# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results(results_dir):
    data = {}
    for model in MODELS:
        path = os.path.join(results_dir, model, 'geometry_results.json')
        if os.path.exists(path):
            with open(path) as f:
                data[model] = json.load(f)
            print(f'  Loaded: {model}')
        else:
            print(f'  Missing: {path}')
    return data


def get_langs(result):
    """Return sorted list of languages present in cosine_similarities."""
    keys = result['cosine_similarities'].keys()
    jb_langs = sorted(set(k.split('_vs_')[0] for k in keys),
                      key=lambda l: ALL_LANGS.index(l) if l in ALL_LANGS else 99)
    return jb_langs


def build_avg_matrix(result, jb_langs, ref_langs):
    """Build matrix[i,j] = avg cosine sim for jb_langs[i] vs ref_langs[j]."""
    cos_sims = result['cosine_similarities']
    mat = np.full((len(jb_langs), len(ref_langs)), np.nan)
    for i, jl in enumerate(jb_langs):
        for j, rl in enumerate(ref_langs):
            key = f'{jl}_vs_{rl}'
            if key in cos_sims:
                mat[i, j] = float(np.mean(cos_sims[key]))
    return mat


def get_resource_level(lang):
    """Return 'LRL' for low-resource languages, 'HRL' for others."""
    return 'LRL' if lang in LRL_LANGS else 'HRL'


def build_cross_matrix(result, langs):
    """Build symmetric matrix of cross-lingual jailbreak vector similarities."""
    cross = result.get('jailbreak_cross_lingual', {})
    n = len(langs)
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 1.0)
    for i, l1 in enumerate(langs):
        for j, l2 in enumerate(langs):
            if i == j:
                continue
            key1 = f'{l1}_vs_{l2}'
            key2 = f'{l2}_vs_{l1}'
            val = None
            if key1 in cross:
                val = cross[key1]['mid_layer']
            elif key2 in cross:
                val = cross[key2]['mid_layer']
            if val is not None:
                mat[i, j] = val
                mat[j, i] = val
    return mat


# ── Figure 1: RQ1 heatmaps ────────────────────────────────────────────────────

def plot_rq1_heatmaps(data, output_dir):
    """One heatmap per model: jailbreak lang × refusal lang, avg cosine sim."""
    available = [m for m in MODELS if m in data]
    fig, axes = plt.subplots(1, len(available),
                             figsize=(6.5 * len(available), 5.5))
    if len(available) == 1:
        axes = [axes]

    vmin, vmax = -0.75, 0.20

    for ax, model in zip(axes, available):
        result  = data[model]
        langs   = get_langs(result)
        mat     = build_avg_matrix(result, langs, langs)

        im = ax.imshow(mat, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(langs)))
        ax.set_yticks(range(len(langs)))
        ax.set_xticklabels(langs, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(langs, fontsize=9)
        ax.set_xlabel('Refusal direction language', fontsize=10)
        ax.set_ylabel('Jailbreak vector language', fontsize=10)
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight='bold')

        # Annotate cells
        for i in range(len(langs)):
            for j in range(len(langs)):
                val = mat[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 0.4 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=6.5, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='Avg cosine similarity')

    fig.suptitle('RQ1: Cosine similarity between jailbreak vectors and refusal directions',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_rq1_heatmaps.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 2: Layer-wise profiles ─────────────────────────────────────────────

def plot_layer_profiles(data, output_dir):
    """Layer-wise cosine similarity for same-language pairs across models."""
    # Pick languages that appear in all models
    lang_sets = [set(get_langs(data[m])) for m in MODELS if m in data]
    common = sorted(lang_sets[0].intersection(*lang_sets[1:]),
                    key=lambda l: ALL_LANGS.index(l) if l in ALL_LANGS else 99)
    common = [l for l in common if l != 'yo'][:6]  # top 6 non-yo languages

    n_cols = min(2, len(common))
    n_rows = (len(common) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, lang in zip(axes, common):
        for model in MODELS:
            if model not in data:
                continue
            key = f'{lang}_vs_{lang}'
            sims = data[model]['cosine_similarities'].get(key)
            if sims is None:
                continue
            layers = list(range(len(sims)))
            ax.plot(layers, sims,
                    label=MODEL_LABELS[model],
                    color=COLORS[model], linewidth=1.8)

        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_title(f'jb={lang} vs refusal={lang}', fontsize=10)
        ax.set_xlabel('Layer', fontsize=9)
        ax.set_ylabel('Cosine similarity', fontsize=9)
        ax.set_ylim(-1.0, 0.5)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes[len(common):]:
        ax.set_visible(False)

    fig.suptitle('RQ1: Layer-wise cosine similarity (same-language pairs)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_rq1_layer_profiles.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 3: RQ2 cross-lingual heatmaps ──────────────────────────────────────

def plot_rq2_heatmaps(data, output_dir):
    """Cross-lingual jailbreak vector similarity at middle layer, per model."""
    available = [m for m in MODELS if m in data]
    fig, axes = plt.subplots(1, len(available),
                             figsize=(5.5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, model in zip(axes, available):
        result = data[model]
        langs  = get_langs(result)
        mat    = build_cross_matrix(result, langs)

        im = ax.imshow(mat, cmap='RdYlBu', vmin=-0.2, vmax=1.0, aspect='auto')
        ax.set_xticks(range(len(langs)))
        ax.set_yticks(range(len(langs)))

        # Color tick labels by resource level: blue = HRL, red = LRL
        x_ticks = ax.set_xticklabels(langs, rotation=45, ha='right', fontsize=9)
        y_ticks = ax.set_yticklabels(langs, fontsize=9)
        for tick, lang in zip(x_ticks, langs):
            tick.set_color(RESOURCE_COLORS[get_resource_level(lang)])
        for tick, lang in zip(y_ticks, langs):
            tick.set_color(RESOURCE_COLORS[get_resource_level(lang)])

        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight='bold')

        for i in range(len(langs)):
            for j in range(len(langs)):
                val = mat[i, j]
                if not np.isnan(val):
                    color = 'white' if val > 0.7 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=6.5, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='Cosine similarity (mid layer)')

        # Mean similarity by resource group pair
        groups = {'HRL–HRL': [], 'HRL–LRL': [], 'LRL–LRL': []}
        for i, l1 in enumerate(langs):
            for j, l2 in enumerate(langs):
                if i >= j:
                    continue
                val = mat[i, j]
                if np.isnan(val):
                    continue
                r1, r2 = get_resource_level(l1), get_resource_level(l2)
                key = 'HRL–HRL' if r1 == r2 == 'HRL' else \
                      'LRL–LRL' if r1 == r2 == 'LRL' else 'HRL–LRL'
                groups[key].append(val)

        lines = []
        for grp, vals in groups.items():
            lines.append(f'{grp}: {np.mean(vals):.2f}  (n={len(vals)})' if vals else f'{grp}: —')
        ax.text(0.02, 0.02, '\n'.join(lines),
                transform=ax.transAxes, fontsize=7.5, va='bottom', ha='left',
                bbox=dict(boxstyle='round', facecolor='white',
                          alpha=0.88, edgecolor='gray', linewidth=0.8))

    fig.suptitle(
        'RQ1 (Geometric): Cross-lingual jailbreak vector similarity\n'
        'blue = HRL,  red = LRL (low-resource, minimal safety alignment)',
        fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_rq2_cross_lingual.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 4: Three-model diagonal comparison ─────────────────────────────────

def plot_diagonal_comparison(data, output_dir):
    """Bar chart: same-language cosine similarity by language, grouped by model."""
    # Collect diagonal values
    records = {}
    for model in MODELS:
        if model not in data:
            continue
        result = data[model]
        langs  = get_langs(result)
        for lang in langs:
            key = f'{lang}_vs_{lang}'
            sims = result['cosine_similarities'].get(key)
            if sims is not None:
                records.setdefault(lang, {})[model] = float(np.mean(sims))

    # Sort languages by ALL_LANGS order
    langs = sorted(records.keys(),
                   key=lambda l: ALL_LANGS.index(l) if l in ALL_LANGS else 99)

    x      = np.arange(len(langs))
    n_mod  = sum(1 for m in MODELS if m in data)
    width  = 0.25
    offset = np.linspace(-(n_mod-1)/2, (n_mod-1)/2, n_mod) * width

    fig, ax = plt.subplots(figsize=(13, 4.5))

    for k, model in enumerate([m for m in MODELS if m in data]):
        vals = [records[l].get(model, np.nan) for l in langs]
        ax.bar(x + offset[k], vals, width,
               label=MODEL_LABELS[model],
               color=COLORS[model], alpha=0.85, edgecolor='white')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=10)
    ax.set_ylabel('Avg cosine similarity', fontsize=11)
    ax.set_title('Same-language cosine similarity (jailbreak vector vs refusal direction), three models',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(-0.85, 0.25)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig4_diagonal_comparison.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 5: HRL vs LRL cosine similarity profiles ──────────────────────────

def plot_yoruba_anomaly(data, output_dir):
    """Layer-wise cos(j_l, r_l) comparing HRL languages vs LRL languages."""
    hrl_examples = ['ko', 'zh']   # representative HRL
    lrl_examples = [l for l in ['yo', 'sw', 'am'] if any(
        f'{l}_vs_{l}' in data[m]['cosine_similarities'] for m in MODELS if m in data
    )]
    show_langs = hrl_examples + lrl_examples
    n_cols = len(show_langs)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, lang in zip(axes, show_langs):
        for model in MODELS:
            if model not in data:
                continue
            key  = f'{lang}_vs_{lang}'
            sims = data[model]['cosine_similarities'].get(key)
            if sims is None:
                continue
            ax.plot(list(range(len(sims))), sims,
                    label=MODEL_LABELS[model],
                    color=COLORS[model], linewidth=2)

        ax.axhline(0, color='gray', linewidth=1, linestyle='--')
        level = 'LRL' if lang in LRL_LANGS else 'HRL'
        color = RESOURCE_COLORS[level]
        ax.set_title(f'{lang}  [{level}]', fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel('Layer', fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel('cos(j_l, r_l)', fontsize=10)
        ax.set_ylim(-1.0, 0.6)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('RQ1: cos(j_l, r_l) layer profiles — HRL vs LRL languages\n'
                 'LRL: j_l not anti-parallel to r_l → jailbreak ≠ suppressing refusal',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig5_hrl_vs_lrl_profiles.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 6: L2 norm of jailbreak vector per layer ──────────────────────────

def plot_l2_norm_profiles(data, output_dir):
    """Layer-wise L2 norm of jailbreak vector for common languages across models."""
    # Collect languages that have l2_norms in at least one model
    lang_sets = []
    for model in MODELS:
        if model not in data:
            continue
        jb_vecs = data[model].get('jailbreak_vectors', {})
        langs_with_norms = {l for l, v in jb_vecs.items() if 'l2_norms' in v}
        if langs_with_norms:
            lang_sets.append(langs_with_norms)

    if not lang_sets:
        print('  Skipping fig6: no l2_norms found (re-run extract_jailbreak_vectors.py)')
        return

    common = sorted(lang_sets[0].intersection(*lang_sets[1:]) if len(lang_sets) > 1 else lang_sets[0],
                    key=lambda l: ALL_LANGS.index(l) if l in ALL_LANGS else 99)
    common = [l for l in common if l != 'yo'][:6]

    if not common:
        print('  Skipping fig6: no common languages with l2_norms.')
        return

    n_cols = min(2, len(common))
    n_rows = (len(common) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, lang in zip(axes, common):
        for model in MODELS:
            if model not in data:
                continue
            jb_info = data[model].get('jailbreak_vectors', {}).get(lang, {})
            norms = jb_info.get('l2_norms')
            if norms is None:
                continue
            layers = list(range(len(norms)))
            best_l = jb_info.get('best_detect_layer')
            ax.plot(layers, norms,
                    label=MODEL_LABELS[model],
                    color=COLORS[model], linewidth=1.8)
            if best_l is not None:
                ax.axvline(best_l, color=COLORS[model],
                           linewidth=1, linestyle=':', alpha=0.7)

        ax.set_title(f'lang={lang}', fontsize=10)
        ax.set_xlabel('Layer', fontsize=9)
        ax.set_ylabel('L2 norm', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(common):]:
        ax.set_visible(False)

    fig.suptitle('Jailbreak direction strength: layer-wise L2 norm\n'
                 '(dotted line = best detection layer per model)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig6_l2_norm_profiles.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 7: Three-way geometric relationship ───────────────────────────────

def plot_three_way(data, output_dir):
    """
    Layer-wise cosine similarity for all three direction pairs per language:
      jb_vec vs refusal_dir | jb_vec vs harmfulness_dir | refusal_dir vs harmfulness_dir
    One subplot per language, lines coloured by pair type, one panel per model.
    """
    PAIR_COLORS = {
        'jb_vs_refusal':          '#E53935',
        'jb_vs_harmfulness':      '#8E24AA',
        'refusal_vs_harmfulness': '#1E88E5',
    }
    PAIR_LABELS = {
        'jb_vs_refusal':          'JB vs Refusal',
        'jb_vs_harmfulness':      'JB vs Harmfulness',
        'refusal_vs_harmfulness': 'Refusal vs Harmfulness',
    }

    # Collect languages that have three_way data in any model
    all_langs = set()
    for model in MODELS:
        if model not in data:
            continue
        for lang in data[model].get('three_way_similarities', {}):
            all_langs.add(lang)

    if not all_langs:
        print('  Skipping fig7: no three_way_similarities found.')
        return

    available = [m for m in MODELS if m in data]

    # Only include languages present in ALL models to avoid blank subplots
    model_lang_sets = [
        set(data[m].get('three_way_similarities', {}).keys())
        for m in available
    ]
    common_langs = model_lang_sets[0].intersection(*model_lang_sets[1:]) if len(model_lang_sets) > 1 else model_lang_sets[0]
    langs = sorted(common_langs, key=lambda l: ALL_LANGS.index(l) if l in ALL_LANGS else 99)
    langs = [l for l in langs if l != 'yo'][:6]
    n_cols = len(langs)
    n_rows = len(available)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.2 * n_rows),
                             sharey=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for r, model in enumerate(available):
        three_way = data[model].get('three_way_similarities', {})
        for c, lang in enumerate(langs):
            ax = axes[r][c]
            if lang not in three_way:
                ax.set_visible(False)
                continue
            tw = three_way[lang]
            for pair, color in PAIR_COLORS.items():
                sims = tw.get(pair)
                if sims is None:
                    continue
                ax.plot(range(len(sims)), sims,
                        color=color, linewidth=1.6,
                        label=PAIR_LABELS[pair])
            ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.set_ylim(-1.0, 1.0)
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(lang, fontsize=10, fontweight='bold')
            if c == 0:
                ax.set_ylabel(MODEL_LABELS[model], fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel('Layer', fontsize=8)
            if r == 0 and c == n_cols - 1:
                ax.legend(fontsize=7, loc='lower right')

    fig.suptitle('Three-way geometric relationship: JB vector / Refusal direction / Harmfulness direction',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig7_three_way_geometry.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 8: RQ2 behavioural transfer results ────────────────────────────────

def plot_transfer_results(transfer_dir, output_dir):
    """
    Grouped bar chart: bypass rate by target language and source jailbreak vector.
    One subplot per model. Baseline shown as a dashed horizontal line per language.
    Only models for which transfer_results.json exists are plotted.
    """
    TARGET_ORDER = ['en', 'de', 'es', 'fr', 'it']
    SOURCE_COLORS = {'yo': '#E53935', 'ko': '#FB8C00', 'ja': '#8E24AA'}
    SOURCE_LABELS = {'yo': 'Yoruba (yo)', 'ko': 'Korean (ko)', 'ja': 'Japanese (ja)'}

    # Load transfer data
    transfer_data = {}
    for model in MODELS:
        path = os.path.join(transfer_dir, model, 'transfer_results.json')
        if os.path.exists(path):
            with open(path) as f:
                transfer_data[model] = json.load(f)
            print(f'  Loaded transfer: {model}')
        else:
            print(f'  Missing transfer: {path}')

    if not transfer_data:
        print('  Skipping fig8: no transfer_results.json found.')
        return

    available = [m for m in MODELS if m in transfer_data]
    fig, axes = plt.subplots(1, len(available),
                             figsize=(6 * len(available), 4.5),
                             sharey=False)
    if len(available) == 1:
        axes = [axes]

    for ax, model in zip(axes, available):
        res = transfer_data[model]['results']
        src_langs = transfer_data[model]['source_langs']
        alpha     = transfer_data[model]['alpha']
        k_star    = transfer_data[model]['k_star']

        # Collect data for each target language (only those present in results)
        tgt_langs = [l for l in TARGET_ORDER if l in res]
        x = np.arange(len(tgt_langs))
        n_src = len(src_langs)
        width = 0.18
        offsets = np.linspace(-(n_src - 1) / 2, (n_src - 1) / 2, n_src) * width

        for k, src in enumerate(src_langs):
            bypass_vals = [res[tgt]['transfer'].get(src, np.nan) for tgt in tgt_langs]
            bars = ax.bar(x + offsets[k], bypass_vals, width,
                          label=SOURCE_LABELS.get(src, src),
                          color=SOURCE_COLORS.get(src, '#607D8B'),
                          alpha=0.85, edgecolor='white')

        # Baseline as scatter points
        baselines = [res[tgt]['baseline_bypass'] for tgt in tgt_langs]
        ax.scatter(x, baselines, marker='D', s=40, color='black',
                   zorder=5, label='Baseline')

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(tgt_langs, fontsize=10)
        ax.set_xlabel('Target language', fontsize=10)
        ax.set_ylabel('Bypass rate', fontsize=10)
        ax.set_title(f'{MODEL_LABELS[model]}\n'
                     f'$\\alpha$={alpha}, $k^*$={k_star}',
                     fontsize=10, fontweight='bold')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle('RQ2 (Behavioural): Cross-lingual jailbreak transfer bypass rates',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig8_rq2_transfer_bypass.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 9: RQ3 cross-lingual defense compliance rates ─────────────────────

def plot_cross_defense_bypass(defense_dir, output_dir):
    """
    Fig 9: Compliance rate matrix after -j_l defense (defense_results.json).
    Input: bypassed samples (baseline=100%). Lower = better defense.
    """
    defense_data = {}
    for model in MODELS:
        path = os.path.join(defense_dir, model, 'defense_results.json')
        if os.path.exists(path):
            with open(path) as f:
                defense_data[model] = json.load(f)
            print(f'  Loaded defense: {model}')
        else:
            print(f'  Missing defense: {path}')

    if not defense_data:
        print('  Skipping fig9: no defense_results.json found.')
        return

    available = [m for m in MODELS if m in defense_data]
    fig, axes = plt.subplots(1, len(available),
                             figsize=(6 * len(available), 5), squeeze=False)

    for col, model in enumerate(available):
        ax       = axes[0][col]
        d        = defense_data[model]
        defended = d['defended']
        tgt_langs = [l for l in d['target_langs'] if l in defended]
        src_langs = d['jb_srcs']

        mat = np.full((len(tgt_langs), len(src_langs)), np.nan)
        for i, tgt in enumerate(tgt_langs):
            for j, src in enumerate(src_langs):
                val = defended.get(tgt, {}).get(src)
                if val is not None:
                    mat[i, j] = val

        im = ax.imshow(mat, cmap='RdYlGn_r', vmin=0.0, vmax=1.0, aspect='auto')
        ax.set_xticks(range(len(src_langs)))
        ax.set_yticks(range(len(tgt_langs)))
        ax.set_xticklabels(src_langs, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(tgt_langs, fontsize=9)

        for tick, lang in zip(ax.get_xticklabels(), src_langs):
            tick.set_color(RESOURCE_COLORS[get_resource_level(lang)])
        for tick, lang in zip(ax.get_yticklabels(), tgt_langs):
            tick.set_color(RESOURCE_COLORS[get_resource_level(lang)])

        ax.set_xlabel('Source j_l language', fontsize=10)
        ax.set_ylabel('Target language (defended)', fontsize=10)
        ax.set_title(f'{MODEL_LABELS[model]}\nα={d["alpha"]}, k*={d["k_star"]}',
                     fontsize=10, fontweight='bold')

        for i in range(len(tgt_langs)):
            for j in range(len(src_langs)):
                val = mat[i, j]
                if not np.isnan(val):
                    is_self = src_langs[j] == tgt_langs[i]
                    label = f'[{val:.2f}]' if is_self else f'{val:.2f}'
                    color = 'white' if val > 0.6 else 'black'
                    ax.text(j, i, label, ha='center', va='center', fontsize=7,
                            color=color, fontweight='bold' if is_self else 'normal')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='Compliance rate (lower = better)')

    fig.suptitle('RQ3: Cross-lingual -j_l defense — compliance rate matrix\n'
                 'Baseline=100% (bypassed samples).  [x.xx]=self-defense.  '
                 'blue=HRL,  red=LRL',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig9_defense_compliance.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 10: RQ3 HRL vs LRL defense analysis ───────────────────────────────

def plot_defense_taxonomy(defense_dir, output_dir):
    """
    Fig 10: Defense effectiveness by resource level.
    Left  — Mean compliance after defense per source language (HRL vs LRL).
    Right — Scatter: cos(j_l, r_l) at k* vs mean compliance rate.
    """
    defense_data = {}
    geo_data = {}
    for model in MODELS:
        dp = os.path.join(defense_dir, model, 'defense_results.json')
        if os.path.exists(dp):
            with open(dp) as f:
                defense_data[model] = json.load(f)
        # Try to find geometry results
        for candidate in [
            os.path.join(defense_dir, '..', 'jailbreak_analysis', model, 'geometry_results.json'),
            os.path.join(defense_dir.replace('defense', 'jailbreak_analysis'), model, 'geometry_results.json'),
        ]:
            gp = os.path.normpath(candidate)
            if os.path.exists(gp):
                with open(gp) as f:
                    geo_data[model] = json.load(f)
                break

    if not defense_data:
        print('  Skipping fig10: no defense_results.json found.')
        return

    available = [m for m in MODELS if m in defense_data]
    rows = []
    for model in available:
        d        = defense_data[model]
        defended = d['defended']
        k_star   = d.get('k_star', 0)
        tgt_list = [l for l in d.get('target_langs', []) if l in defended]

        cos_j_r_per_lang = {}
        if model in geo_data:
            cos_sims = geo_data[model].get('cosine_similarities', {})
            for lang in d.get('jb_srcs', []):
                key = f'{lang}_vs_{lang}'
                if key in cos_sims:
                    cos_j_r_per_lang[lang] = cos_sims[key][k_star]

        for src in d.get('jb_srcs', []):
            comp_vals = [defended.get(tgt, {}).get(src)
                         for tgt in tgt_list
                         if defended.get(tgt, {}).get(src) is not None]
            if not comp_vals:
                continue
            rows.append({
                'model':     model,
                'src':       src,
                'resource':  get_resource_level(src),
                'mean_comp': float(np.mean(comp_vals)),
                'cos_j_r':   cos_j_r_per_lang.get(src, float('nan')),
            })

    if not rows:
        print('  Skipping fig10: no data.')
        return

    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(12, 4.5))

    all_srcs = sorted(set(r['src'] for r in rows),
                      key=lambda l: (get_resource_level(l), l))
    x       = np.arange(len(all_srcs))
    n_m     = len(available)
    width   = 0.22
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

    for k, model in enumerate(available):
        model_map = {r['src']: r['mean_comp'] for r in rows if r['model'] == model}
        vals = [model_map.get(s, np.nan) for s in all_srcs]
        ax_bar.bar(x + offsets[k], vals, width, label=MODEL_LABELS[model],
                   color=COLORS[model], alpha=0.85, edgecolor='white', linewidth=0.5)

    ax_bar.axhline(1.0, color='black', linewidth=1, linestyle='--',
                   alpha=0.5, label='Baseline (100%)')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [f'{s}\n[{get_resource_level(s)}]' for s in all_srcs], fontsize=9)
    for tick, src in zip(ax_bar.get_xticklabels(), all_srcs):
        tick.set_color(RESOURCE_COLORS[get_resource_level(src)])
    ax_bar.set_ylabel('Mean compliance rate after defense', fontsize=10)
    ax_bar.set_title('Defense by source language\n(lower = better)', fontsize=10)
    ax_bar.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, axis='y', alpha=0.3)
    ax_bar.set_ylim(0, 1.1)

    for r in rows:
        if np.isnan(r['cos_j_r']):
            continue
        ax_scatter.scatter(
            r['cos_j_r'], r['mean_comp'],
            color=COLORS[r['model']],
            marker='o' if r['resource'] == 'HRL' else 's',
            s=90, zorder=3, edgecolors='white', linewidth=0.5)
        ax_scatter.annotate(r['src'], xy=(r['cos_j_r'], r['mean_comp']),
                            xytext=(5, 3), textcoords='offset points', fontsize=7)

    ax_scatter.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax_scatter.axvline(0, color='gray', linewidth=0.8, linestyle=':', alpha=0.4)
    ax_scatter.set_xlabel('$\\cos(\\hat{j}_l,\\, \\hat{r}_l)$ at $k^*$', fontsize=10)
    ax_scatter.set_ylabel('Mean compliance after defense', fontsize=10)
    ax_scatter.set_title('cos(j_l, r_l) vs defense effectiveness\n'
                         'circle=HRL,  square=LRL', fontsize=10)
    ax_scatter.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.set_ylim(0, 1.1)

    valid = [(r['cos_j_r'], r['mean_comp']) for r in rows if not np.isnan(r['cos_j_r'])]
    if len(valid) >= 3:
        cos_v, comp_v = zip(*valid)
        r_val = float(np.corrcoef(cos_v, comp_v)[0, 1])
        ax_scatter.text(0.05, 0.95, f'Pearson $r$ = {r_val:.2f}',
                        transform=ax_scatter.transAxes, fontsize=10, va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[m],
                   markersize=8, label=MODEL_LABELS[m])
        for m in available
    ] + [
        plt.Line2D([0], [0], marker='o', color='gray', markersize=8, label='HRL source'),
        plt.Line2D([0], [0], marker='s', color='gray', markersize=8, label='LRL source'),
    ]
    ax_scatter.legend(handles=legend_handles, fontsize=7, loc='lower right')

    fig.suptitle('RQ3: HRL vs LRL as defense source — cos(j_l,r_l) predicts effectiveness',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig10_defense_hrl_lrl.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading results...')
    data = load_results(args.results_dir)

    if not data:
        print('No results found. Check --results_dir.')
        return

    print('\nGenerating figures...')
    plot_rq1_heatmaps(data, args.output_dir)
    plot_layer_profiles(data, args.output_dir)
    plot_rq2_heatmaps(data, args.output_dir)
    plot_diagonal_comparison(data, args.output_dir)
    plot_yoruba_anomaly(data, args.output_dir)
    plot_l2_norm_profiles(data, args.output_dir)
    plot_three_way(data, args.output_dir)
    plot_transfer_results(args.transfer_dir, args.output_dir)
    plot_cross_defense_bypass(args.defense_dir, args.output_dir)
    plot_defense_taxonomy(args.defense_dir, args.output_dir)

    print(f'\nDone. All figures saved to {args.output_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str,
                        default='output/jailbreak_analysis',
                        help='Directory containing per-model geometry_results.json files')
    parser.add_argument('--transfer_dir', type=str,
                        default='output/transfer',
                        help='Directory containing per-model transfer_results.json files')
    parser.add_argument('--defense_dir', type=str,
                        default='output/defense',
                        help='Directory containing per-model defense_results.json files')
    parser.add_argument('--output_dir', type=str,
                        default='output/figures',
                        help='Where to save figures')
    args = parser.parse_args()
    main(args)
