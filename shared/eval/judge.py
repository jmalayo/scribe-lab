import re

VERDICT_RE = re.compile(r"TRUE|FALSE", re.IGNORECASE)

def parse_verdict(raw: str | None) -> tuple[bool, bool]:

    match = VERDICT_RE.search(raw or "")

    if not match:
        print(f"  [!] Veredicto no reconocido, se toma como FALSE: {(raw or '')[:50]!r}", flush=True)

        return False, False

    return match.group().upper() == "TRUE", True

def agreement_rate(verdicts_a: list[dict], verdicts_b: list[dict], key: str) -> float:

    if not verdicts_a:
        return 0.0

    matches = 0

    for a, b in zip(verdicts_a, verdicts_b, strict=True):
        if a[key] == b[key]:
            matches += 1

    return round(matches / len(verdicts_a), 4)
