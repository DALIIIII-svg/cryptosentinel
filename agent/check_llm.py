"""Quick check that the configured LLM answers with valid JSON (make llm-check)."""
from __future__ import annotations

import sys
import time

from agent import llm
from common.config import get_settings


def main() -> int:
    s = get_settings()
    print(f"provider={s.llm_provider} model={s.llm_model} key={'set' if s.groq_api_key else 'EMPTY'}")
    t0 = time.time()
    try:
        out = llm.complete("Answer with one JSON object only.", 'Return {"status": "ok", "model_name": "<your name>"}')
        print(f"answer in {time.time() - t0:.1f} s: {out[:200]}")
        print("parsed:", llm.extract_json(out))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
