"""Push the built dataset to the Hugging Face Hub.

Uploads both configs as JSONL plus the dataset card in data/dataset/README.md.
Requires `huggingface-cli login` (or HF_TOKEN) beforehand.

Usage:
  python -m pipeline.hf_upload --repo <user-or-org>/xlcost-variable-roles \
      --dataset data/dataset [--private]
"""

import argparse
import os

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--card", default="data/dataset/README.md")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    user = api.whoami()
    print(f"Logged in as: {user['name']}")

    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=args.card, path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")
    for config in ("python_perturbations", "multilingual_baseline"):
        folder = os.path.join(args.dataset, config)
        api.upload_folder(folder_path=folder, path_in_repo=config,
                          repo_id=args.repo, repo_type="dataset",
                          commit_message=f"Upload {config}")
    stats = os.path.join(args.dataset, "stats.json")
    if os.path.exists(stats):
        api.upload_file(path_or_fileobj=stats, path_in_repo="stats.json",
                        repo_id=args.repo, repo_type="dataset")
    print(f"Done: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
