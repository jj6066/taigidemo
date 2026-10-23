# Breeze-ASR-26 FP16 模型

這個 repository 主要分享 **Breeze-ASR-26 的 FP16 權重**。模型檔位於 `models/Breeze-ASR-26-fp16/`；`taigi.py` 和 `breeze_asr.py` 是用來示範載入模型與辨識音檔的程式。

此 FP16 權重是 [MediaTek-Research/Breeze-ASR-26](https://huggingface.co/MediaTek-Research/Breeze-ASR-26) 的 FP16 版本。上游模型採用 Apache-2.0 授權，請參閱上游模型頁面的授權與使用條款。

## 下載模型

權重使用 Git LFS 儲存。在 Ubuntu 安裝 Git LFS，然後下載 `fp16-model` 分支：

```bash
sudo apt update
sudo apt install git-lfs
git lfs install
git clone --branch fp16-model https://github.com/jj6066/taigidemo.git
cd taigidemo
git lfs pull
```

GitHub Free／Pro 的 Git LFS 單檔上限是 2 GB。第一個權重 shard 約 2.50 GB，因此拆成兩段存放；第二個 shard 保持單一檔案。下載後執行以下指令重組第一個 shard：

```bash
bash scripts/reassemble_fp16.sh
```

重組後的模型目錄包含 Transformers 載入所需的兩個完整權重檔：

```text
models/Breeze-ASR-26-fp16/model-00001-of-00002.safetensors
models/Breeze-ASR-26-fp16/model-00002-of-00002.safetensors
```

完整權重約 3.1 GB（2.9 GiB）。GitHub Free／Pro 每月包含 10 GiB LFS 儲存空間與 10 GiB 下載流量；下載模型會使用 repository 擁有者的流量額度。

## 示範程式

`taigi.py` 接受音檔路徑並輸出台語語音辨識結果。Ubuntu 需先安裝 FFmpeg：

```bash
sudo apt install ffmpeg
```

建立 Python 虛擬環境並安裝依賴：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

執行示範：

```bash
python taigi.py path/to/audio.m4a
```

若使用 NVIDIA GPU，請依照 [PyTorch 官方安裝說明](https://pytorch.org/get-started/locally/)安裝符合 CUDA 環境的 PyTorch。
