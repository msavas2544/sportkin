"""Table export that opens cleanly in SPSS, R, jamovi and (Turkish) Excel."""

from __future__ import annotations

import csv
import math
from pathlib import Path

__all__ = ["write_table", "read_columns"]


def _fmt(v, decimal: str, digits: int) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool,)):
        return str(int(v))
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return ""  # SPSS/R treat empty cells as missing
    s = f"{f:.{digits}f}"
    return s.replace(".", ",") if decimal == "," else s


def write_table(rows, path, *, decimal: str = ".", digits: int = 4, columns=None) -> Path:
    """Write a list of dicts (or a dict of columns) to CSV.

    ``decimal="."`` -> comma-separated (SPSS, R, jamovi default).
    ``decimal=","`` -> semicolon-separated with decimal commas (Turkish Excel).
    Missing values are written as empty cells.
    """
    if isinstance(rows, dict):  # column-oriented
        cols = list(rows)
        n = len(next(iter(rows.values()))) if rows else 0
        rows = [{c: rows[c][i] for c in cols} for i in range(n)]
    rows = list(rows)
    if columns is None:
        columns = list(rows[0]) if rows else []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    delim = ";" if decimal == "," else ","
    with path.open("w", newline="", encoding="utf-8-sig") as fh:  # BOM for Excel
        w = csv.writer(fh, delimiter=delim)
        w.writerow(columns)
        for r in rows:
            w.writerow([_fmt(r.get(c), decimal, digits) for c in columns])
    return path


def read_columns(path, *names):
    """Read numeric columns from a CSV (comma or semicolon, '.' or ',' decimals)."""
    text = Path(path).read_text(encoding="utf-8-sig")
    delim = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delim)
    rows = list(reader)
    missing = [n for n in names if n not in reader.fieldnames]
    if missing:
        raise KeyError(f"columns not found: {missing}; available: {reader.fieldnames}")
    out = []
    for n in names:
        col = []
        for r in rows:
            v = (r[n] or "").strip().replace(",", ".")
            col.append(float(v) if v else float("nan"))
        out.append(col)
    return out
