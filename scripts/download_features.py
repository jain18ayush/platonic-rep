"""Download precomputed features from HuggingFace Hub."""

from huggingface_hub import snapshot_download

snapshot_download(
    "jain18ayush/platonic-rep-features",
    repo_type="dataset",
    local_dir="results/features",
)
print("Features downloaded to results/features/")
