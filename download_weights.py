"""Download pre-extracted C weights from HuggingFace.

Fetches one or more c_weights_{mode}/ directories from
https://huggingface.co/eschmidbauer/parakeet-unified-en-0.6b-c
so you can skip the ONNX conversion step in extract_weights.py.

Usage:
    python download_weights.py fp32        # download c_weights_fp32/
    python download_weights.py int8        # download c_weights_int8/
    python download_weights.py mixed       # download c_weights_mixed/
    python download_weights.py             # download all three
"""
import argparse
import os
import sys

from huggingface_hub import snapshot_download

HF_REPO = "eschmidbauer/parakeet-unified-en-0.6b-c"
MODES = ("fp32", "int8", "mixed")


def download_mode(mode):
    subdir = f"c_weights_{mode}"
    expected = {"weights.bin", "weights.json", "tokenizer.model"}
    if os.path.isdir(subdir) and expected.issubset(set(os.listdir(subdir))):
        print(f"{subdir}/ already present, skipping download")
        return

    print(f"downloading {subdir}/ from {HF_REPO} …")
    snapshot_download(
        repo_id=HF_REPO,
        allow_patterns=f"{subdir}/*",
        local_dir=".",
    )
    print(f"{subdir}/ ready")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download C weights from HuggingFace")
    parser.add_argument("mode", nargs="?", choices=MODES, default=None,
                        help="weight format to download (default: all)")
    args = parser.parse_args()

    for mode in ([args.mode] if args.mode else MODES):
        download_mode(mode)
