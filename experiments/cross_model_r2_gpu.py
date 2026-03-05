"""
Full-scale cross-model R² analysis across all val-set models.
Three stages: extract, analyze, plot.

Usage:
    python experiments/cross_model_r2_gpu.py extract --modality language
    python experiments/cross_model_r2_gpu.py extract --modality vision
    python experiments/cross_model_r2_gpu.py extract --modality language --models "bigscience/bloomz-560m"
    python experiments/cross_model_r2_gpu.py analyze
    python experiments/cross_model_r2_gpu.py plot
"""

import gc
import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange, tqdm

# Add repo root to path so we can import from top-level modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor
from datasets import load_dataset

from tasks import get_models
from models import load_llm, load_tokenizer
from metrics import remove_outliers
from measure_alignment import prepare_features, compute_score
import utils


# ---------------------------------------------------------------------------
# Cross-model R²
# ---------------------------------------------------------------------------

def cross_model_r2(X, Y, n_folds=5):
    """Cross-validated R² for predicting Y from X via ridge regression.

    Args:
        X: torch tensor of shape (N, D1) — predictor features
        Y: torch tensor of shape (N, D2) — target features
        n_folds: number of CV folds

    Returns:
        Mean cross-validated R² score.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
    Y_np = Y.cpu().numpy() if Y.is_cuda else Y.numpy()
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 10)))
    scores = cross_val_score(model, X_np, Y_np, cv=n_folds, scoring="r2")
    return scores.mean()


# ---------------------------------------------------------------------------
# Stage 1: Extract features (identical to intrinsic_dimensionality_gpu.py)
# ---------------------------------------------------------------------------

DATASET_NAME = "minhuh/prh"
SUBSET = "wit_1024"
FEATURE_DIR = "./results/features"
ANALYSIS_DIR = "./results/cross_model_r2"


def extract_language(models, batch_size=4):
    """Extract features for language models."""
    dataset = load_dataset(DATASET_NAME, revision=SUBSET, split="train")
    texts = [str(x["text"][0]) for x in dataset]

    for model_name in models:
        save_path = utils.to_feature_filename(
            FEATURE_DIR, DATASET_NAME, SUBSET, model_name, pool="avg"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            print(f"[skip] {save_path} already exists")
            continue

        print(f"\n{'='*60}")
        print(f"Extracting: {model_name}")
        print(f"Save path:  {save_path}")

        language_model = load_llm(model_name)
        param_count = sum(p.numel() for p in language_model.parameters())
        tokenizer = load_tokenizer(model_name)

        tokens = tokenizer(texts, padding="longest", return_tensors="pt")
        device = next(language_model.parameters()).device

        all_feats = []
        for i in trange(0, len(dataset), batch_size, desc=model_name):
            token_inputs = {k: v[i : i + batch_size].to(device).long() for k, v in tokens.items()}
            with torch.no_grad():
                output = language_model(
                    input_ids=token_inputs["input_ids"],
                    attention_mask=token_inputs["attention_mask"],
                )
                # avg pool over sequence length
                feats = torch.stack(output["hidden_states"]).permute(1, 0, 2, 3)
                mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)
                feats = (feats * mask).sum(2) / mask.sum(2)
                all_feats.append(feats.cpu())

        save_dict = {"feats": torch.cat(all_feats).cpu(), "num_params": param_count}
        torch.save(save_dict, save_path)
        print(f"Saved {save_path}  shape={save_dict['feats'].shape}  params={param_count:,}")

        del language_model, tokenizer, all_feats
        torch.cuda.empty_cache()
        gc.collect()


def extract_vision(models, batch_size=32):
    """Extract features for vision models."""
    dataset = load_dataset(DATASET_NAME, revision=SUBSET, split="train")

    for model_name in models:
        save_path = utils.to_feature_filename(
            FEATURE_DIR, DATASET_NAME, SUBSET, model_name, pool="cls"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            print(f"[skip] {save_path} already exists")
            continue

        print(f"\n{'='*60}")
        print(f"Extracting: {model_name}")
        print(f"Save path:  {save_path}")

        vision_model = timm.create_model(model_name, pretrained=True).cuda().eval()
        param_count = sum(p.numel() for p in vision_model.parameters())
        transform = create_transform(
            **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
        )

        return_nodes = [f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))]
        vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)

        all_feats = []
        for i in trange(0, len(dataset), batch_size, desc=model_name):
            with torch.no_grad():
                ims = torch.stack(
                    [transform(dataset[j]["image"]) for j in range(i, min(i + batch_size, len(dataset)))]
                ).cuda()
                output = vision_model(ims)
                feats = torch.stack([v[:, 0, :] for v in output.values()]).permute(1, 0, 2)
                all_feats.append(feats.cpu())

        save_dict = {"feats": torch.cat(all_feats).cpu(), "num_params": param_count}
        torch.save(save_dict, save_path)
        print(f"Saved {save_path}  shape={save_dict['feats'].shape}  params={param_count:,}")

        del vision_model, transform, all_feats
        torch.cuda.empty_cache()
        gc.collect()


def run_extract(args):
    llm_models, lvm_models = get_models("val", modality="all")

    if args.modality in ("language", "all"):
        models = llm_models
        if args.models:
            models = [m for m in args.models.split(",") if m in llm_models]
            if not models:
                print(f"Warning: none of the specified models found in language model list")
        extract_language(models, batch_size=args.batch_size or 4)

    if args.modality in ("vision", "all"):
        models = lvm_models
        if args.models:
            models = [m for m in args.models.split(",") if m in lvm_models]
            if not models:
                print(f"Warning: none of the specified models found in vision model list")
        extract_vision(models, batch_size=args.batch_size or 32)


# ---------------------------------------------------------------------------
# Stage 2: Analyze — compute R² at best-aligning layer pairs
# ---------------------------------------------------------------------------

def run_analyze(args):
    llm_models, lvm_models = get_models("val", modality="all")

    # --- Collect available models and their metadata ---
    model_info = {}

    for modality, model_list, pool in [("language", llm_models, "avg"), ("vision", lvm_models, "cls")]:
        for model_name in model_list:
            feat_path = utils.to_feature_filename(
                FEATURE_DIR, DATASET_NAME, SUBSET, model_name, pool=pool
            )
            if not os.path.exists(feat_path):
                print(f"[skip] features not found: {feat_path}")
                continue

            data = torch.load(feat_path, map_location="cpu")
            num_params = int(data["num_params"])
            num_layers = data["feats"].shape[1]

            model_info[model_name] = {
                "num_params": num_params,
                "modality": modality,
                "num_layers": num_layers,
            }
            print(f"{model_name}: params={num_params:,}  layers={num_layers}")
            del data

    available_llms = [m for m in llm_models if m in model_info]
    available_lvms = [m for m in lvm_models if m in model_info]

    total_pairs = len(available_llms) * len(available_lvms)
    print(f"\nAnalyzing {total_pairs} cross-modal pairs...")

    # --- Two-pass strategy ---
    # Pass 1: Use mutual_knn to find best-aligning layer pair for each cross-modal pair
    # Pass 2: Fix vision layer at best, sweep language layers for R²

    pair_analysis = []

    for lang_model in tqdm(available_llms, desc="LLM"):
        lang_path = utils.to_feature_filename(
            FEATURE_DIR, DATASET_NAME, SUBSET, lang_model, pool="avg"
        )
        raw_lang = torch.load(lang_path, map_location="cuda:0")["feats"].float()
        lang_feats_prepared = prepare_features(raw_lang, exact=True)
        num_lang_layers = raw_lang.shape[1]

        for vision_model in tqdm(available_lvms, desc="  ViT", leave=False):
            vision_path = utils.to_feature_filename(
                FEATURE_DIR, DATASET_NAME, SUBSET, vision_model, pool="cls"
            )
            raw_vision = torch.load(vision_path, map_location="cuda:0")["feats"].float()
            vision_feats_prepared = prepare_features(raw_vision, exact=True)

            # Pass 1: find best-aligning layers via mutual_knn
            alignment_score, best_layers = compute_score(
                lang_feats_prepared, vision_feats_prepared,
                metric="mutual_knn", topk=10,
            )
            lang_best_layer, vis_best_layer = best_layers

            # Pass 2: compute R² with vision layer fixed at best
            # Preprocess features for R² (remove outliers + normalize, on CPU)
            vis_feats_at_best = raw_lang.new_zeros(1)  # placeholder
            vf = raw_vision[:, vis_best_layer, :].cpu()
            vf = remove_outliers(vf, q=0.95)
            vf = F.normalize(vf, p=2, dim=-1)

            r2_fwd_per_lang_layer = []
            r2_rev_per_lang_layer = []

            for lang_layer in range(num_lang_layers):
                lf = raw_lang[:, lang_layer, :].cpu()
                lf = remove_outliers(lf, q=0.95)
                lf = F.normalize(lf, p=2, dim=-1)

                r2_fwd = cross_model_r2(lf, vf)
                r2_rev = cross_model_r2(vf, lf)
                r2_fwd_per_lang_layer.append(float(r2_fwd))
                r2_rev_per_lang_layer.append(float(r2_rev))

            # R² at the best-aligning layer pair
            r2_fwd_at_best = r2_fwd_per_lang_layer[lang_best_layer]
            r2_rev_at_best = r2_rev_per_lang_layer[lang_best_layer]
            r2_mean_at_best = (r2_fwd_at_best + r2_rev_at_best) / 2

            pair_analysis.append({
                "lang_model": lang_model,
                "vision_model": vision_model,
                "best_layers": [lang_best_layer, vis_best_layer],
                "alignment_score": float(alignment_score),
                "r2_forward_at_best": r2_fwd_at_best,
                "r2_reverse_at_best": r2_rev_at_best,
                "r2_mean_at_best": r2_mean_at_best,
                "r2_forward_per_lang_layer": r2_fwd_per_lang_layer,
                "r2_reverse_per_lang_layer": r2_rev_per_lang_layer,
            })

            print(f"  {lang_model} x {vision_model}: "
                  f"align={alignment_score:.3f}  "
                  f"R²_fwd={r2_fwd_at_best:.3f}  R²_rev={r2_rev_at_best:.3f}  "
                  f"R²_mean={r2_mean_at_best:.3f}")

            del vision_feats_prepared, raw_vision
            torch.cuda.empty_cache()

        del lang_feats_prepared, raw_lang
        torch.cuda.empty_cache()

    # Save results
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    out_path = os.path.join(ANALYSIS_DIR, "analysis.json")
    with open(out_path, "w") as f:
        json.dump({"models": model_info, "pair_analysis": pair_analysis}, f, indent=2)
    print(f"\nSaved analysis to {out_path}")


# ---------------------------------------------------------------------------
# Stage 3: Plot
# ---------------------------------------------------------------------------

def run_plot(args):
    import matplotlib.pyplot as plt

    analysis_path = os.path.join(ANALYSIS_DIR, "analysis.json")
    with open(analysis_path) as f:
        data = json.load(f)

    models = data["models"]
    pairs = data["pair_analysis"]

    # --- Family classification for coloring ---
    def get_family(name):
        if "bloom" in name.lower():
            return "BLOOM"
        elif "open_llama" in name.lower():
            return "OpenLLaMA"
        elif "llama" in name.lower():
            return "LLaMA"
        return "Other"

    family_colors = {"BLOOM": "#1f77b4", "OpenLLaMA": "#ff7f0e", "LLaMA": "#2ca02c", "Other": "#7f7f7f"}

    # ===== Figure A: R² vs Model Scale =====
    llm_names = [m for m, info in models.items() if info["modality"] == "language"]
    llm_avg = {}
    for llm in llm_names:
        llm_pairs = [p for p in pairs if p["lang_model"] == llm]
        if not llm_pairs:
            continue
        llm_avg[llm] = {
            "num_params": models[llm]["num_params"],
            "avg_r2": np.mean([p["r2_mean_at_best"] for p in llm_pairs]),
            "avg_alignment": np.mean([p["alignment_score"] for p in llm_pairs]),
            "family": get_family(llm),
        }

    fig, ax = plt.subplots(figsize=(10, 6))
    for llm, info in llm_avg.items():
        color = family_colors.get(info["family"], "#7f7f7f")
        size = 50 + 200 * info["avg_alignment"]
        ax.scatter(
            info["num_params"], info["avg_r2"],
            c=color, s=size, alpha=0.8, edgecolors="k", linewidths=0.5, zorder=3,
        )
        ax.annotate(
            llm.split("/")[-1], (info["num_params"], info["avg_r2"]),
            fontsize=7, ha="left", va="bottom", xytext=(5, 3), textcoords="offset points",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters", fontsize=12)
    ax.set_ylabel("Avg Cross-Validated R² at Best-Aligning Layers", fontsize=12)
    ax.set_title("Cross-Modal R² vs Model Scale", fontsize=14)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=fam)
        for fam, c in family_colors.items() if fam != "Other"
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_a_path = os.path.join(ANALYSIS_DIR, "fig_r2_vs_scale.png")
    fig.savefig(fig_a_path, dpi=150)
    print(f"Saved {fig_a_path}")
    plt.close(fig)

    # ===== Figure B: R² Across Language Model Layers =====
    fig, ax = plt.subplots(figsize=(12, 6))

    # Average R² across all vision models for each LLM
    for llm in llm_names:
        llm_pairs = [p for p in pairs if p["lang_model"] == llm]
        if not llm_pairs:
            continue

        num_lang_layers = len(llm_pairs[0]["r2_forward_per_lang_layer"])
        avg_fwd = np.mean([p["r2_forward_per_lang_layer"] for p in llm_pairs], axis=0)
        avg_rev = np.mean([p["r2_reverse_per_lang_layer"] for p in llm_pairs], axis=0)

        x = np.linspace(0, 1, num_lang_layers)
        family = get_family(llm)
        color = family_colors.get(family, "#7f7f7f")
        short_name = llm.split("/")[-1]

        ax.plot(x, avg_fwd, "-", color=color, alpha=0.7, linewidth=1.2, label=f"{short_name} (fwd)")
        ax.plot(x, avg_rev, "--", color=color, alpha=0.7, linewidth=1.2, label=f"{short_name} (rev)")

    ax.set_xlabel("Normalized Language Model Layer Depth", fontsize=12)
    ax.set_ylabel("Cross-Validated R²", fontsize=12)
    ax.set_title("R² Across Language Model Layers (averaged over vision models)", fontsize=14)
    ax.legend(fontsize=5, ncol=3, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_b_path = os.path.join(ANALYSIS_DIR, "fig_r2_across_layers.png")
    fig.savefig(fig_b_path, dpi=150, bbox_inches="tight")
    print(f"Saved {fig_b_path}")
    plt.close(fig)

    # ===== Figure C: R² vs Alignment Score =====
    fig, ax = plt.subplots(figsize=(8, 6))

    for p in pairs:
        family = get_family(p["lang_model"])
        color = family_colors.get(family, "#7f7f7f")
        ax.scatter(
            p["alignment_score"], p["r2_mean_at_best"],
            c=color, alpha=0.6, s=40, edgecolors="k", linewidths=0.3,
        )

    ax.set_xlabel("Mutual KNN Alignment Score", fontsize=12)
    ax.set_ylabel("Cross-Validated R² (mean fwd+rev)", fontsize=12)
    ax.set_title("R² vs Alignment Score (each dot = one LLM-ViT pair)", fontsize=14)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=fam)
        for fam, c in family_colors.items() if fam != "Other"
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_c_path = os.path.join(ANALYSIS_DIR, "fig_r2_vs_alignment.png")
    fig.savefig(fig_c_path, dpi=150)
    print(f"Saved {fig_c_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-model R² analysis (GPU)")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract per-layer features for models")
    p_extract.add_argument("--modality", type=str, default="all", choices=["language", "vision", "all"])
    p_extract.add_argument("--models", type=str, default=None,
                           help="Comma-separated model names to extract (subset of val set)")
    p_extract.add_argument("--batch_size", type=int, default=None,
                           help="Batch size (default: 4 for language, 32 for vision)")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Compute R² for cross-modal pairs")

    # plot
    p_plot = subparsers.add_parser("plot", help="Generate figures from analysis results")

    args = parser.parse_args()

    if args.stage == "extract":
        run_extract(args)
    elif args.stage == "analyze":
        run_analyze(args)
    elif args.stage == "plot":
        run_plot(args)


if __name__ == "__main__":
    main()
