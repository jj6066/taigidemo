# 台語語音辨識

使用 MediaTek Breeze-ASR-26 將音檔轉成文字。程式支援命令列輸入音檔，也提供可供其他 Python 程式呼叫的轉錄函式。

## 環境需求

- Python 3.10 或更新版本
- FFmpeg
- Breeze-ASR-26 模型檔（首次使用前下載，約數 GB）

Ubuntu 安裝 FFmpeg：

```bash
sudo apt update
sudo apt install ffmpeg
```

## 安裝

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

若要使用 NVIDIA GPU，請依照 [PyTorch 官方安裝指引](https://pytorch.org/get-started/locally/)安裝符合 CUDA 環境的 PyTorch。

## 下載模型

模型權重不放在這個 Git 專案中。從 [MediaTek-Research/Breeze-ASR-26](https://huggingface.co/MediaTek-Research/Breeze-ASR-26) 下載模型檔至程式預期的資料夾：

```bash
python -m pip install -U huggingface_hub
hf download MediaTek-Research/Breeze-ASR-26 --local-dir models/Breeze-ASR-26-fp16
```

下載完成後，目錄中應包含 `config.json`、`model.safetensors.index.json` 和模型權重檔。模型權重很大，下載需要時間與數 GB 的磁碟空間。

## 使用

```bash
python taigi.py path/to/audio.m4a
```

不帶路徑執行時，程式會提示輸入音檔路徑：

```bash
python taigi.py
```

也可以在 Python 程式中呼叫：

```python
from breeze_asr import transcribe_audio_file

text = transcribe_audio_file("path/to/audio.m4a")
print(text)
```

`breeze_asr.py` 也提供 `transcribe_audio_bytes()`，可轉錄記憶體中的音訊資料，例如服務下載到的 LINE 語音內容。
