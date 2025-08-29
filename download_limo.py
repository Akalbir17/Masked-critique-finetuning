from huggingface_hub import snapshot_download
from datasets import load_dataset

# Specify repository ID and cache directory
repo_id = "GAIR/LIMO"
cache_dir = "/project2/jieyuz_1540/.cache/huggingface/datasets"

# Download the entire dataset repository
snapshot_download(repo_id=repo_id, repo_type="dataset", cache_dir=cache_dir)

print("Dataset downloaded successfully!")


# Load the dataset from Hugging Face and cache it in your specified directory
ds = load_dataset("GAIR/LIMO", cache_dir="/project2/jieyuz_1540/.cache/huggingface/datasets")

# Check the first training example
print("Dataset loaded successfully!")
print(ds["train"][0])