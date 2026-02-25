"""Smoke test: verify all 4 OpenRouter models respond correctly.

Usage:
    OPENROUTER_API_KEY=sk-or-... python benchmarks/model_adapter_smoke_test.py

Each model is asked "What is 2+2? Reply with just the number."
A PASS requires the response to contain "4".
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, "/home/keith/contextprune")

from contextprune.adapters import SUPPORTED_MODELS, CompletionResult, OpenRouterAdapter


def get_api_key() -> str:
    """Get OpenRouter API key from env or 1Password."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key

    try:
        result = subprocess.run(
            [
                "bash",
                "-c",
                "OP_SERVICE_ACCOUNT_TOKEN=$(cat /etc/op-service-account-token) "
                "op item get 'OPENROUTER_API_KEY' --vault 'Keef Secrets' "
                "--fields credential --reveal 2>/dev/null",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        key = result.stdout.strip()
        if key:
            return key
    except Exception as exc:
        print(f"Warning: could not fetch key from 1Password: {exc}", file=sys.stderr)

    sys.exit("ERROR: OPENROUTER_API_KEY not set and 1Password fetch failed.")


PROMPT = "What is 2+2? Reply with just the number."


def main() -> None:
    api_key = get_api_key()
    adapter = OpenRouterAdapter(api_key=api_key)

    col_w = {"alias": 8, "model": 40, "latency": 12, "cost": 14, "resp": 20, "status": 6}
    header = (
        f"{'Alias':<{col_w['alias']}} "
        f"{'Model':<{col_w['model']}} "
        f"{'Latency (ms)':>{col_w['latency']}} "
        f"{'Cost (USD)':>{col_w['cost']}} "
        f"{'Response':<{col_w['resp']}} "
        f"{'Pass?':<{col_w['status']}}"
    )
    sep = "-" * len(header)

    print("\nSmoke Test — all 4 OpenRouter models")
    print(sep)
    print(header)
    print(sep)

    passed = 0
    failed = 0

    for alias, model_id in SUPPORTED_MODELS.items():
        try:
            result: CompletionResult = adapter.complete(
                messages=[{"role": "user", "content": PROMPT}],
                model=alias,
                max_tokens=16,
                temperature=0.0,
            )
            ok = "4" in result.text
            status = "✓ PASS" if ok else "✗ FAIL"
            if ok:
                passed += 1
            else:
                failed += 1

            resp_preview = result.text.strip().replace("\n", " ")[:col_w["resp"]]
            print(
                f"{alias:<{col_w['alias']}} "
                f"{model_id:<{col_w['model']}} "
                f"{result.latency_ms:>{col_w['latency']}.1f} "
                f"{result.cost_usd:>{col_w['cost']}.8f} "
                f"{resp_preview:<{col_w['resp']}} "
                f"{status:<{col_w['status']}}"
            )
        except Exception as exc:
            failed += 1
            err = str(exc)[:40]
            print(
                f"{alias:<{col_w['alias']}} "
                f"{model_id:<{col_w['model']}} "
                f"{'N/A':>{col_w['latency']}} "
                f"{'N/A':>{col_w['cost']}} "
                f"{'ERROR: ' + err:<{col_w['resp']}} "
                f"{'✗ FAIL':<{col_w['status']}}"
            )

    print(sep)
    print(f"\nResults: {passed} passed, {failed} failed out of {len(SUPPORTED_MODELS)} models.")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
