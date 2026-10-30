"""Make a de-identified copy of the ARIAS subject sheet that Claude Code may read.

YOU run this on your own machine. It never prints cell values, only column names and
counts, so it is safe to run inside a Claude Code session too.

    python scripts/deidentify_arias.py ~/leelab/ARIAS_subjects.csv
    python scripts/deidentify_arias.py ~/leelab/ARIAS_subjects.xlsx --sheet 0

Keep the RAW sheet outside this repo (the repo is public). The output goes to
private/arias_subjects_deid.csv, which is gitignored.

What is kept (matched by column name, case-insensitive):
    subject   study code like 101-1234 (needed to join with image files)
    group     LR / HR / MCI
    sex
    age       whole years. Taken from an age column, or computed from DOB and
              collection/visit date. DOB itself is never written out.
    quality_left, quality_right   image quality scores
    eye_left, eye_right           1 if that eye has an OCTA scan listed, else 0
    exclude, exclude_reason       empty columns for YOU to fill in by hand
Everything else (names, DOB, dates, uid/eid/did device IDs, filenames, free-text notes)
is dropped. Free-text notes are dropped on purpose: they can mention other subjects.
Use --show-columns first if the matching looks wrong, then --map to fix it.
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "private" / "arias_subjects_deid.csv"

# output column -> regex tried against normalized input column names, in order
PATTERNS = {
    "subject": [r"^subject(_?id)?$", r"^subj(ect)?", r"^id$", r"^participant"],
    "group": [r"^group$", r"^group", r"^cohort", r"^arm$", r"^dx$", r"^diagnosis"],
    "sex": [r"^sex$", r"^gender$"],
    "age": [r"^age$", r"^age_?(y|yr|years)?$"],
    "quality_left": [r"quality.*(left|_l$|\bos\b)", r"(left|os).*quality"],
    "quality_right": [r"quality.*(right|_r$|\bod\b)", r"(right|od).*quality"],
}
DOB_PAT = [r"^dob$", r"birth"]
VISIT_PAT = [r"collect", r"visit", r"exam", r"scan_?date", r"^date$"]
OCTA_LEFT_PAT = [r"octa.*(left|_l$|\bos\b)", r"(left|os).*octa"]
OCTA_RIGHT_PAT = [r"octa.*(right|_r$|\bod\b)", r"(right|od).*octa"]


def norm(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def find(cols, patterns, exclude=()):
    for pat in patterns:
        for c in cols:
            if c not in exclude and re.search(pat, norm(c)):
                return c
    return None


def to_date(series):
    """Accept date strings or Excel serial numbers, even mixed in one column."""
    num = pd.to_numeric(series, errors="coerce")
    from_serial = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_text = pd.to_datetime(series.where(num.isna()), errors="coerce")
    return from_serial.fillna(from_text)


def main():
    ap = argparse.ArgumentParser(description="De-identify the ARIAS subject sheet.")
    ap.add_argument("raw", type=Path, help="raw CSV or XLSX (keep it outside the repo)")
    ap.add_argument("--sheet", default=0, help="sheet index or name for .xlsx")
    ap.add_argument("--show-columns", action="store_true", help="list column names and exit")
    ap.add_argument("--map", nargs="*", default=[],
                    help="override matching, e.g. --map group='Risk Group' sex=Gender")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    raw = args.raw.expanduser().resolve()
    if REPO in raw.parents:
        print(f"NOTE: the raw sheet is inside the repo ({raw}). Move it outside, e.g. ~/leelab/.")
    if raw.suffix.lower() in (".xlsx", ".xls"):
        sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
        df = pd.read_excel(raw, sheet_name=sheet, dtype=str)
    else:
        df = pd.read_csv(raw, dtype=str, keep_default_na=False)
    df = df.dropna(how="all")
    cols = list(df.columns)

    if args.show_columns:
        print(f"{len(df)} rows. Columns:")
        for c in cols:
            print("  ", repr(c))
        return

    overrides = dict(m.split("=", 1) for m in args.map)
    chosen = {}
    for out_col, pats in PATTERNS.items():
        src = overrides.get(out_col) or find(cols, pats, exclude=chosen.values())
        if src is not None and src not in cols:
            sys.exit(f"--map {out_col}={src}: no such column")
        chosen[out_col] = src

    out = pd.DataFrame(index=df.index)
    for out_col, src in chosen.items():
        out[out_col] = df[src].str.strip() if src else ""

    if not chosen["subject"]:
        sys.exit("Could not find a subject column. Run with --show-columns, then --map subject=...")
    bad_ids = ~out["subject"].str.fullmatch(r"\d{3}-\d{4}")
    if bad_ids.any():
        print(f"WARNING: {int(bad_ids.sum())} subject values do not look like ddd-dddd; "
              "they were blanked. Check the raw sheet by hand.")
        out.loc[bad_ids, "subject"] = ""

    dob_col = overrides.get("dob") or find(cols, DOB_PAT)
    visit_col = overrides.get("visit") or find(cols, VISIT_PAT, exclude=[dob_col])
    age_from = chosen["age"] or "(none)"
    if dob_col and visit_col:
        age = (to_date(df[visit_col]) - to_date(df[dob_col])).dt.days / 365.25
        computed = age.round(0).astype("Int64").astype(str).replace("<NA>", "")
        if chosen["age"]:
            out["age"] = out["age"].where(out["age"] != "", computed)
            age_from = f"{chosen['age']} (gaps filled from DOB and visit date)"
        else:
            out["age"] = computed
            age_from = "computed from DOB and visit date"
    else:
        out["age"] = pd.to_numeric(out["age"], errors="coerce").round(0).astype("Int64").astype(str).replace("<NA>", "")

    for side, pats in (("left", OCTA_LEFT_PAT), ("right", OCTA_RIGHT_PAT)):
        src = overrides.get(f"eye_{side}") or find(cols, pats, exclude=chosen.values())
        out[f"eye_{side}"] = ((df[src].fillna("").str.strip() != "").astype(int) if src else "")
        chosen[f"eye_{side}"] = src

    out["exclude"] = ""
    out["exclude_reason"] = ""
    out = out[out["subject"] != ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    kept = {c for c in chosen.values() if c} | {c for c in (dob_col, visit_col) if c}
    print(f"Wrote {len(out)} rows to {args.out}")
    print("Column mapping (input -> output):")
    for k, v in chosen.items():
        print(f"   {k:14s} <- {v!r}")
    print(f"   {'age source':14s}    {age_from}")
    used = [c for c in (dob_col, visit_col) if c]
    if used:
        print("Used only to compute age, NOT written out:", used)
    print("Dropped columns:", [c for c in cols if c not in kept])
    if "group" in out:
        print("Group counts:", out["group"].value_counts().to_dict())
    print("Next: open the file yourself, fill in exclude / exclude_reason, and check no names slipped in.")


if __name__ == "__main__":
    main()
