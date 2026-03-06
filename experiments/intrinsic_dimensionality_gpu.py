"""
Full-scale intrinsic dimensionality analysis across all val-set models.
Three stages: extract, analyze, plot.

Usage:
    python experiments/intrinsic_dimensionality_gpu.py extract --modality language
    python experiments/intrinsic_dimensionality_gpu.py extract --modality vision
    python experiments/intrinsic_dimensionality_gpu.py extract --modality language --models "bigscience/bloomz-560m,bigscience/bloomz-1b1"
    python experiments/intrinsic_dimensionality_gpu.py analyze
    python experiments/intrinsic_dimensionality_gpu.py plot
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
import utils


# ---------------------------------------------------------------------------
# Participation ratio
# ---------------------------------------------------------------------------

def participation_ratio(feats):
    """Compute participation ratio (effective dimensionality) of feature matrix."""
    cov = torch.cov(feats.T)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.clamp(min=0)
    return (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()


# ---------------------------------------------------------------------------
# Stage 1: Extract features
# ---------------------------------------------------------------------------

DATASET_NAME = "minhuh/prh"
SUBSET = "wit_1024"
FEATURE_DIR = "./results/features"
ANALYSIS_DIR = "./results/intrinsic_dimensionality"


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
# Stage 2: Analyze
# ---------------------------------------------------------------------------

def fast_mutual_knn(knn_a, knn_b, n, topk):
    """Mutual KNN from precomputed KNN indices — no matmul needed."""
    range_tensor = torch.arange(n, device=knn_a.device).unsqueeze(1)
    mask_a = torch.zeros(n, n, device=knn_a.device)
    mask_b = torch.zeros(n, n, device=knn_b.device)
    mask_a[range_tensor, knn_a] = 1.0
    mask_b[range_tensor, knn_b] = 1.0
    return ((mask_a * mask_b).sum(dim=1) / topk).mean().item()


def run_analyze(args):
    llm_models, lvm_models = get_models("val", modality="all")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    topk = 10

    # --- Per-model analysis: load features, compute PR per layer ---
    # Also precompute KNN indices for each layer to avoid redundant computation
    model_info = {}
    model_knn = {}  # model_name -> list of KNN index tensors per layer

    for modality, model_list, pool in [("language", llm_models, "avg"), ("vision", lvm_models, "cls")]:
        for model_name in model_list:
            feat_path = utils.to_feature_filename(
                FEATURE_DIR, DATASET_NAME, SUBSET, model_name, pool=pool
            )
            if not os.path.exists(feat_path):
                print(f"[skip] features not found: {feat_path}")
                continue

            data = torch.load(feat_path, map_location="cpu")
            feats = data["feats"].float()  # (N, L, D)
            num_params = int(data["num_params"])
            num_layers = feats.shape[1]

            pr_per_layer = []
            knn_per_layer = []
            for layer_idx in range(num_layers):
                layer_feats = feats[:, layer_idx, :].to(device)
                # remove outliers + normalize (same preprocessing as alignment)
                layer_feats = remove_outliers(layer_feats, q=0.95, exact=True)
                layer_feats_norm = F.normalize(layer_feats, p=2, dim=-1)

                # PR computation on GPU
                pr = participation_ratio(layer_feats_norm).item()
                pr_per_layer.append(pr)

                # Precompute KNN indices for this layer
                knn = (layer_feats_norm @ layer_feats_norm.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
                knn_per_layer.append(knn)  # keep on GPU

                del layer_feats, layer_feats_norm

            model_info[model_name] = {
                "num_params": num_params,
                "modality": modality,
                "pr_per_layer": pr_per_layer,
                "pr_last_layer": pr_per_layer[-1],
            }
            model_knn[model_name] = knn_per_layer
            print(f"{model_name}: params={num_params:,}  layers={num_layers}  PR_last={pr_per_layer[-1]:.1f}")

            del data, feats

    # --- Pair analysis: cross-modal alignment using precomputed KNN ---
    pair_analysis = []
    available_llms = [m for m in llm_models if m in model_info]
    available_lvms = [m for m in lvm_models if m in model_info]

    total_pairs = len(available_llms) * len(available_lvms)
    print(f"\nComputing alignment for {total_pairs} cross-modal pairs...")

    for lang_model in tqdm(available_llms, desc="LLM"):
        lang_knns = model_knn[lang_model]
        n = lang_knns[0].shape[0]

        for vision_model in available_lvms:
            vision_knns = model_knn[vision_model]

            best_score = 0
            best_layers = (0, 0)

            for i, knn_lang in enumerate(lang_knns):
                for j, knn_vis in enumerate(vision_knns):
                    score = fast_mutual_knn(knn_lang, knn_vis, n=n, topk=topk)
                    if score > best_score:
                        best_score = score
                        best_layers = (i, j)

            lang_layer_idx, vision_layer_idx = best_layers

            pr_lang = model_info[lang_model]["pr_per_layer"][lang_layer_idx]
            pr_vision = model_info[vision_model]["pr_per_layer"][vision_layer_idx]
            dim_gap = abs(pr_vision - pr_lang)
            dim_gap_last = abs(
                model_info[vision_model]["pr_last_layer"]
                - model_info[lang_model]["pr_last_layer"]
            )

            pair_analysis.append({
                "lang_model": lang_model,
                "vision_model": vision_model,
                "alignment_score": best_score,
                "best_layers": [lang_layer_idx, vision_layer_idx],
                "pr_at_best_lang": pr_lang,
                "pr_at_best_vision": pr_vision,
                "dim_gap_at_best": dim_gap,
                "dim_gap_last_layer": dim_gap_last,
            })

    # Free KNN cache
    del model_knn
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

    # ===== Figure A: Dimensionality Gap vs Scale =====
    # For each LLM, average dim gap and alignment across all vision pairs
    llm_names = [m for m, info in models.items() if info["modality"] == "language"]
    llm_avg = {}
    for llm in llm_names:
        llm_pairs = [p for p in pairs if p["lang_model"] == llm]
        if not llm_pairs:
            continue
        llm_avg[llm] = {
            "num_params": models[llm]["num_params"],
            "avg_dim_gap": np.mean([p["dim_gap_at_best"] for p in llm_pairs]),
            "avg_alignment": np.mean([p["alignment_score"] for p in llm_pairs]),
            "family": get_family(llm),
        }

    fig, ax = plt.subplots(figsize=(10, 6))
    for llm, info in llm_avg.items():
        color = family_colors.get(info["family"], "#7f7f7f")
        size = 50 + 200 * info["avg_alignment"]  # scale dot by alignment
        ax.scatter(
            info["num_params"], info["avg_dim_gap"],
            c=color, s=size, alpha=0.8, edgecolors="k", linewidths=0.5, zorder=3,
        )
        ax.annotate(
            llm.split("/")[-1], (info["num_params"], info["avg_dim_gap"]),
            fontsize=7, ha="left", va="bottom", xytext=(5, 3), textcoords="offset points",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters", fontsize=12)
    ax.set_ylabel("Avg Dimensionality Gap (PR) at Best-Aligning Layers", fontsize=12)
    ax.set_title("Intrinsic Dimensionality Gap vs Model Scale", fontsize=14)

    # Legend for families
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=fam)
        for fam, c in family_colors.items() if fam != "Other"
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_a_path = os.path.join(ANALYSIS_DIR, "fig_dim_gap_vs_scale.png")
    fig.savefig(fig_a_path, dpi=150)
    print(f"Saved {fig_a_path}")
    plt.close(fig)

    # ===== Figure B: PR Across Layers =====
    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name, info in models.items():
        pr = info["pr_per_layer"]
        num_layers = len(pr)
        x = np.linspace(0, 1, num_layers)

        is_vision = info["modality"] == "vision"
        linestyle = "-" if is_vision else "--"
        family = get_family(model_name) if not is_vision else "Vision"

        if is_vision:
            color = "#d62728" if "dino" in model_name else "#9467bd" if "clip" in model_name else "#8c564b"
            alpha = 0.5
        else:
            color = family_colors.get(family, "#7f7f7f")
            alpha = 0.7

        ax.plot(x, pr, linestyle=linestyle, color=color, alpha=alpha, linewidth=1.2,
                label=model_name.split("/")[-1])

    ax.set_xlabel("Normalized Layer Depth", fontsize=12)
    ax.set_ylabel("Participation Ratio", fontsize=12)
    ax.set_title("Intrinsic Dimensionality (PR) Across Layers", fontsize=14)
    ax.legend(fontsize=5, ncol=3, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_b_path = os.path.join(ANALYSIS_DIR, "fig_pr_across_layers.png")
    fig.savefig(fig_b_path, dpi=150, bbox_inches="tight")
    print(f"Saved {fig_b_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Intrinsic dimensionality analysis (GPU)")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract per-layer features for models")
    p_extract.add_argument("--modality", type=str, default="all", choices=["language", "vision", "all"])
    p_extract.add_argument("--models", type=str, default=None,
                           help="Comma-separated model names to extract (subset of val set)")
    p_extract.add_argument("--batch_size", type=int, default=None,
                           help="Batch size (default: 4 for language, 32 for vision)")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Compute PR and cross-modal alignment")

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
