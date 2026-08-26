"""Gate calibrate() on verified forward-paper rows (one-shot)."""
import pathlib

p = pathlib.Path("bot/cost_calibration.py")
src = p.read_text(encoding="utf-8")

old_sig = """def calibrate(
    observations: list[dict],
    v1_frictions: dict,
    min_observations: int = 30,
) -> dict:"""
new_sig = """def calibrate(
    observations: list[dict],
    v1_frictions: dict,
    min_observations: int = 30,
    freeze_manifest: dict | None = None,
) -> dict:
    from bot.evidence import verified_forward_rows

    if freeze_manifest is not None:
        observations, exclusions = verified_forward_rows(observations, freeze_manifest)
    else:
        exclusions = {}"""
assert old_sig in src, "sig"
src = src.replace(old_sig, new_sig, 1)

anchor = '    return {\n        "n_turnover_events": n,'
addition = (
    '    return {\n'
    '        "evidence_exclusions": exclusions,\n'
    '        "n_turnover_events": n,'
)
assert anchor in src, "ret"
src = src.replace(anchor, addition, 1)

p.write_text(src, encoding="utf-8")
print("calibrate gated")
