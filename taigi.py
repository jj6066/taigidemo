#!/usr/bin/env python3
"""輸入音檔，呼叫 Breeze-ASR 辨識台語並輸出文字。"""

from pathlib import Path
import os
import site
import sys


def restart_without_user_site() -> None:
    """Keep conda packages from being mixed with incompatible ~/.local ones."""
    if __name__ != "__main__" or not site.ENABLE_USER_SITE:
        return
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    os.execve(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        env,
    )


restart_without_user_site()


def get_audio_path() -> Path:
    """Get a local audio path from the command line or an interactive prompt."""
    if len(sys.argv) > 2:
        raise SystemExit("用法：python taigi.py [音檔路徑]")

    raw_path = sys.argv[1] if len(sys.argv) == 2 else input("請輸入音檔路徑：")
    audio_path = Path(raw_path.strip().strip("\"'")).expanduser()
    if not audio_path.is_file():
        raise SystemExit(f"找不到音檔：{audio_path}")
    return audio_path


def main() -> None:
    audio_path = get_audio_path()
    try:
        from breeze_asr import transcribe_audio_file

        print(f"辨識音檔：{audio_path}")
        text = transcribe_audio_file(audio_path)
    except Exception as exc:
        raise SystemExit(f"語音辨識失敗：{exc}") from exc

    print("\n辨識結果：")
    print(text)


if __name__ == "__main__":
    main()
