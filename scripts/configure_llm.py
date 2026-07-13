from __future__ import annotations

import argparse
from getpass import getpass
from pathlib import Path


def clipboard_text() -> str:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        value = root.clipboard_get()
        root.destroy()
        return value
    except Exception as exc:
        raise RuntimeError("cannot read Windows clipboard; copy the key and try again") from exc


def valid_key(key: str) -> bool:
    return len(key) >= 20 and key.startswith("sk-") and all(character.isprintable() for character in key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Save an ignored local LLM provider configuration")
    parser.add_argument(
        "--from-clipboard", action="store_true", help="read the API key from the Windows clipboard"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    key = (clipboard_text() if args.from_clipboard else getpass("Paste API key (input hidden): ")).strip()
    if not valid_key(key):
        print("The key is invalid or paste was not captured; configuration was not changed.")
        return 2
    base = input("Base URL [https://www.qilinapi.com/v1]: ").strip() or "https://www.qilinapi.com/v1"
    model = input("Model [gpt-5.5]: ").strip() or "gpt-5.5"
    if not base.startswith("https://"):
        print("Base URL must use HTTPS; configuration was not changed.")
        return 2
    destination = root / ".env.local"
    destination.write_text(
        f"OPENAI_API_KEY={key}\nOPENAI_BASE_URL={base.rstrip('/')}\nOPENAI_MODEL={model}\n",
        encoding="utf-8",
    )
    print(
        f"Saved local provider configuration to {destination.name}; key length={len(key)}, value not printed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
