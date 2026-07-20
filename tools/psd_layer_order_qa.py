#!/usr/bin/env python3
"""Layer-ordering QA: verifies the one depth_median ordering invariant
upstream's further_extr actually enforces in code (inference_utils.py
~line 527-534, mirrored -- and fixed to give each tag a distinct offset
instead of upstream's collapse-to-one-value -- in postproc.cpp's crop_part
face clamp): nose/mouth/eye-group tags must sort in FRONT of "face" (smaller
depth_median), ear tags must sort just BEHIND it (larger depth_median).

Everything else (hair vs headwear, topwear vs neck, etc.) is left to
Marigold's raw per-tag depth estimate with no hard-coded invariant upstream
itself enforces -- inventing rules for those here would be checking this
codebase against guesses, not against upstream. Also flags depth_median
ties, which postproc.cpp's own comment says should no longer happen (each
clamp tag gets a distinct epsilon offset specifically to avoid unspecified
tie-breaks downstream).

Usage:
    python tools/psd_layer_order_qa.py <out.psd.json>
"""
import argparse
import json
import sys

# (front_tags, back_tag): every present front_tag must have depth_median <=
# back_tag's (front = smaller depth_median, given pipeline.cpp's descending
# sort puts larger depth_median at the bottom of the back-to-front stack)
FRONT_OF_FACE = [
    "eyewhite", "eyewhite-l", "eyewhite-r",
    "irides", "irides-l", "irides-r",
    "eyelash", "eyelash-l", "eyelash-r",
    "eyebrow", "eyebrow-l", "eyebrow-r",
    "nose", "mouth",
]
BEHIND_FACE = ["ears", "earl", "earr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("psd_json", help="the <out.psd>.json sidecar written alongside a PSD export")
    args = ap.parse_args()

    with open(args.psd_json, encoding="utf-8") as f:
        doc = json.load(f)
    parts = doc["parts"]

    failures = []

    face = parts.get("face")
    if face is not None:
        face_dm = face["depth_median"]
        for tag in FRONT_OF_FACE:
            if tag in parts and parts[tag]["depth_median"] > face_dm:
                failures.append(
                    f"{tag}: depth_median={parts[tag]['depth_median']:.6f} is BEHIND face "
                    f"({face_dm:.6f}) -- should be in front (<=)"
                )
        for tag in BEHIND_FACE:
            if tag in parts and parts[tag]["depth_median"] < face_dm:
                failures.append(
                    f"{tag}: depth_median={parts[tag]['depth_median']:.6f} is IN FRONT of face "
                    f"({face_dm:.6f}) -- should be behind (>=)"
                )
    else:
        print("note: no 'face' layer present -- face-relative checks skipped")

    seen = {}
    for tag, p in parts.items():
        dm = p["depth_median"]
        if dm in seen:
            failures.append(f"{tag}: depth_median={dm:.6f} ties with {seen[dm]} "
                            f"-- tie-break order becomes unspecified downstream")
        else:
            seen[dm] = tag

    print(f"{len(parts)} layers checked")
    if failures:
        print(f"\n{len(failures)} ordering issue(s):")
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)

    print("all ordering invariants hold")


if __name__ == "__main__":
    main()
