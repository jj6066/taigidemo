# Breeze-ASR-26 FP16

This repository shares the FP16-converted weights for [MediaTek-Research/Breeze-ASR-26](https://huggingface.co/MediaTek-Research/Breeze-ASR-26). The Python files in this repository are demo helpers; the model weights are the main artifact.

The upstream model is published under the Apache-2.0 license. See the [upstream model card](https://huggingface.co/MediaTek-Research/Breeze-ASR-26) for its license and usage details.

## Download

The weight files use Git LFS. On Ubuntu, install Git LFS and clone this branch:

```bash
sudo apt update
sudo apt install git-lfs
git lfs install
git clone --branch fp16-model https://github.com/jj6066/taigidemo.git
cd taigidemo
```

The first model shard is split into two files to fit GitHub Free/Pro's 2 GB per-file Git LFS limit. Reassemble it after cloning:

```bash
bash scripts/reassemble_fp16.sh
```

The reconstructed file will be `models/Breeze-ASR-26-fp16/model-00001-of-00002.safetensors`. The second shard is already a single file. The complete weights use about 3.1 GB (2.9 GiB) of Git LFS storage.

## Demo

The helper scripts expect the model under `models/Breeze-ASR-26-fp16` and use FFmpeg to decode audio. On Ubuntu, install FFmpeg with:

```bash
sudo apt install ffmpeg
```

Install the Python dependencies and run the command-line demo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python taigi.py path/to/audio.m4a
```

For NVIDIA GPU use, install the matching CUDA build of PyTorch using the [official PyTorch instructions](https://pytorch.org/get-started/locally/).
