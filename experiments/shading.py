"""Shading of the best, second and third best cells of a table row or column (TMLR tables).

The LaTeX side defines \\rA, \\rB, \\rC (preamble.tex): dark, medium and light background, \\rA also in bold. Ranks are
dense ranks of the values as displayed, so equal displayed values share a shade.
"""


def ranked(values, fmt="{:.3f}", higher_better=False, parse=None):
    """values: list of numbers (None = not ranked, printed as given by `fmt_none`). Returns the formatted cells.
    fmt: format string or callable producing the cell text; parse: callable mapping the cell text back to the number
    that is ranked (default float of the text with LaTeX math stripped)."""
    def text(v):
        return fmt(v) if callable(fmt) else fmt.format(v)

    def num(s):
        if parse is not None:
            return parse(s)
        t = s.replace("$", "").replace("\\,", "").replace("{", "").replace("}", "").replace("+", "").strip()
        t = t.replace("--", "-")
        return float(t)

    shown = [None if v is None else text(v) for v in values]
    keyed = [None if s is None else num(s) for s in shown]
    distinct = sorted({k for k in keyed if k is not None}, reverse=higher_better)
    rank = {v: i + 1 for i, v in enumerate(distinct)}
    out = []
    for s, k in zip(shown, keyed):
        if s is None:
            out.append(None)
            continue
        r = rank[k]
        out.append({1: "\\rA{%s}" % s, 2: "\\rB{%s}" % s, 3: "\\rC{%s}" % s}.get(r, s))
    return out
