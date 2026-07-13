from __future__ import annotations

from getpass import getpass
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    key = getpass("Paste API key (input hidden): ").strip()
    if not key:
        print("No key supplied; configuration was not changed.")
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
    print(f"Saved local provider configuration to {destination.name}; the key was not printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
