"""Write docs/generation_modes.html from the resolver (rate_table, resolve_generation). Nothing in the tables is typed by hand.
Run from SCAPES/: ../venv_SCAPES/bin/python scripts/make_generation_doc.py --out docs/generation_modes.html"""
import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SCAPES.generation import CAP, CODECS, DEFAULT_RATE, atom_geometry, n_max, rate_table, resolve_generation  # noqa: E402

USER = [k for k, v in CODECS.items() if v.user]

CSS = """:root{--mem:#8fb6e8;--tgt:#e8893a;--bg:#fff;--fg:#1a1a1a;--mut:#6a6a6a;--line:#e3e3e3;--head:#f4f4f4;--acc:#e8f0fe;--zero:#d0d0d0}
@media(prefers-color-scheme:dark){:root{--mem:#4f7fb8;--tgt:#e8893a;--bg:#16181b;--fg:#e8e8e8;--mut:#9a9a9a;--line:#2c2f33;--head:#1f2226;--acc:#1d2a3d;--zero:#3a3d42}}
body{font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:2rem auto;max-width:1000px;padding:0 1rem}
h1{font-size:1.4rem} h2{font-size:1.1rem;margin-top:2.2rem}
p,ul{max-width:80ch} code{background:var(--head);padding:0 .25rem;border-radius:3px}
.wrap{overflow:auto;max-height:70vh;border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;white-space:nowrap;width:100%}
th,td{padding:.35rem .8rem;border-bottom:1px solid var(--line);text-align:right}
th{background:var(--head);position:sticky;top:0;font-weight:600}
tr:hover td{background:var(--acc)} td.def,tr.def td{background:var(--acc);font-weight:600}
.grid{display:flex;flex-wrap:wrap;gap:.4rem;padding:.6rem}
.cell{border:1px solid var(--line);border-radius:4px;padding:.2rem .5rem;font-size:12.5px}
.formulas{background:var(--head);border-radius:6px;padding:.6rem 1.2rem;max-width:80ch;line-height:1.9}
.ex{border-left:3px solid var(--tgt);padding:.1rem .8rem;margin:.5rem 0;max-width:80ch}
.diag{width:100%;max-width:980px;height:auto;display:block;margin:1rem 0 2rem}
.diag .h{font:600 15px system-ui,sans-serif;fill:var(--fg)} .diag .t{font:12.5px system-ui,sans-serif;fill:var(--fg)}
.diag .mut{fill:var(--mut)} .diag .mem{fill:var(--mem)} .diag .tgt{fill:var(--tgt)} .diag .zero{fill:var(--zero)}
.diag .ghost{fill:none;stroke:var(--mut);stroke-dasharray:3 3} .diag .br{stroke:var(--mut);stroke-width:1.4;fill:none}
.diag .win{stroke:var(--tgt);stroke-width:2.5;fill:none} .diag .cf{fill:var(--tgt);opacity:.2}
"""


def fmt(v, d=2):
    return f"{v:.{d}f}"


def rect(x, y, w, h, cls):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" class="{cls}"/>'


def text(x, y, s, cls="t", anchor="start"):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{html.escape(s)}</text>'


def svg_autoregressive():
    p = ['<svg viewBox="0 0 980 190" role="img" aria-label="Autoregressive" class="diag">',
         text(10, 22, "Autoregressive", "h"),
         text(10, 40, "one block of n latents per step; the model attends to the memory window W (attended = W)", "t mut")]
    x0, w, gap = 40, 40, 4
    for i in range(14):
        p.append(rect(x0 + i * (w + gap), 90, w, 34, "mem"))
    p.append(rect(x0 + 14 * (w + gap), 90, 2 * w + gap, 34, "tgt"))
    p.append(rect(x0 + 14 * (w + gap) + 2 * (w + gap), 90, 2 * w + gap, 34, "ghost"))
    p.append(f'<path d="M{x0},84 V76 H{x0 + 14 * (w + gap) - gap + 2 * (w + gap)} V84" class="br"/>')
    p.append(text(330, 68, "memory window W latents (at most 300)", "t", "middle"))
    p.append(text(x0 + 14 * (w + gap) + w, 144, "block n", "t", "middle"))
    p.append(text(x0 + 16 * (w + gap) + w + 20, 144, "next block", "t mut", "middle"))
    p.append(text(40, 170, "past latents (memory)", "t mut"))
    p.append(text(430, 170, "each step the window slides by n latents", "t mut"))
    p.append("</svg>")
    return "\n".join(p)


def svg_atoms():
    # atom [o | x | n] with the window over it, then K buffer atoms shifted by n
    p = ['<svg viewBox="0 0 980 360" role="img" aria-label="Atoms" class="diag">',
         text(10, 22, "Atoms", "h"),
         text(10, 40, "one atom of A = o + x + n latents; the window zeroes o, rises over x, stays at one, falls over x", "t mut")]
    u = 40                       # drawn width of one latent
    o, x, n = 4, 2, 4
    ax = 60
    p.append(rect(ax, 60, o * u, 30, "zero"))
    p.append(rect(ax + o * u, 60, x * u, 30, "cf"))
    p.append(rect(ax + (o + x) * u, 60, (n - x) * u, 30, "tgt"))
    p.append(rect(ax + (o + n) * u, 60, x * u, 30, "cf"))
    p.append(text(ax + o * u / 2, 108, "o: zeroed part", "t", "middle"))
    p.append(text(ax + (o + x / 2) * u, 108, "x: crossfade", "t", "middle"))
    p.append(text(ax + (o + x + (n - x) / 2) * u, 108, "new part", "t", "middle"))
    p.append(text(ax + (o + n + x / 2) * u, 108, "x: crossfade", "t", "middle"))
    # window curve (equal power rise and fall)
    base, top = 190, 130
    pts = [f"{ax},{base}", f"{ax + o * u},{base}"]
    for i in range(0, 9):
        t = i / 8
        import math
        pts.append(f"{ax + (o + x * t) * u:.1f},{base - (base - top) * math.sin(math.pi * t / 2):.1f}")
    pts.append(f"{ax + (o + n) * u},{top}")
    for i in range(0, 9):
        t = i / 8
        pts.append(f"{ax + (o + n + x * t) * u:.1f},{base - (base - top) * math.cos(math.pi * t / 2):.1f}")
    p.append(f'<polyline points="{" ".join(pts)}" class="win"/>')
    p.append(text(ax + (o + n + x) * u + 14, 160, "window (equal power)", "t mut"))
    # buffer atoms, each shifted by n latents
    p.append(text(10, 226, "buffer: K past atoms, each shifted by n latents; the newest one is the atom being generated", "t mut"))
    for k in range(3):
        cls = "tgt" if k == 2 else "mem"
        y = 240 + k * 34
        p.append(rect(ax + k * n * u, y, (o + x + n) * u, 26, cls))
        p.append(text(ax + k * n * u + 8, y + 17, "past atom" if k < 2 else "current atom", "t"))
    p.append(text(ax, 350, "attended = K * A latents (at most 300)", "t mut"))
    p.append("</svg>")
    return "\n".join(p)


def codec_summary():
    rows = []
    for c in USER:
        s = CODECS[c]
        xd = s.crossfade_default if s.mode == "atoms" else "-"
        am = s.min_atom if s.mode == "atoms" else "-"
        rows.append(f"<tr><td>{c}</td><td>{s.mode}</td><td>{fmt(s.latent_rate, 3)}</td><td>{xd}</td><td>{am}</td>"
                    f"<td>{s.n_min}</td><td>{n_max(s)}</td><td>{fmt(s.default_memory_seconds, 1)}</td></tr>")
    head = ("<tr><th>codec</th><th>mode</th><th>latent rate (Hz)</th><th>x default</th><th>A min</th><th>n min</th>"
            "<th>n max</th><th>default memory (s)</th></tr>")
    return f'<div class="wrap"><table>{head}{"".join(rows)}</table></div>'


def example(label, codec, rate=None, mem=None):
    r = resolve_generation(codec, rate, mem)
    s = f"{label}: n = {r.n}, {r.rate:.2f} Hz"
    if r.mode == "atoms":
        s += (f"; x = {r.crossfade}, o = {r.overlapped}, A = {r.atom}; memory {r.memory_latents} latents, "
              f"K = {r.memory_atoms} (K max = {r.k_max}); attended {r.attended}")
    else:
        s += f"; W = {r.memory_window}; attended {r.attended}"
    if r.notes:
        s += f"; notes: {', '.join(r.notes)}"
    return f'<div class="ex" data-example="{html.escape(label)}">{html.escape(s)}</div>'


def examples():
    return "\n".join([
        example("encodec48, rate 12 Hz, memory 0.5 s", "encodec48", 12, 0.5),
        example("music2latent, rate 3 Hz, memory 1 s", "music2latent", 3, 1),
        example("spectrostream, rate 5 Hz, memory 2 s", "spectrostream", 5, 2),
        example("music2latent, rate 10 Hz, memory 10 s", "music2latent", 10, 10),
        example("encodec24, rate 20 Hz", "encodec24", 20),
        example("encodec24, rate 12 Hz", "encodec24", 12),
    ])


def defaults_table():
    rows = []
    for c in USER:
        r = resolve_generation(c)
        a = f"{r.crossfade}, {r.overlapped}, {r.atom}" if r.mode == "atoms" else "-"
        k = r.memory_atoms if r.mode == "atoms" else "-"
        w = r.memory_window if r.mode == "autoregressive" else "-"
        rows.append(f"<tr><td>{c}</td><td>{r.n}</td><td>{r.rate:.3f}</td><td>{a}</td><td>{r.memory_latents}</td>"
                    f"<td>{k}</td><td>{w}</td><td>{r.attended}</td><td>{', '.join(r.notes) or '-'}</td></tr>")
    head = ("<tr><th>codec</th><th>n</th><th>rate (Hz)</th><th>x, o, A</th><th>memory (latents)</th><th>K</th><th>W</th>"
            "<th>attended</th><th>notes</th></tr>")
    return f'<div class="wrap"><table>{head}{"".join(rows)}</table></div>'


def codec_table(codec):
    s = CODECS[codec]
    atoms = s.mode == "atoms"
    default_n = resolve_generation(codec).n
    full = [r for r in rate_table(codec) if r["rate_hz"] >= 1]
    slow = [r for r in rate_table(codec) if r["rate_hz"] < 1]
    out = [f"<h3>{codec} ({s.mode}), n = {s.n_min} to {n_max(s)}</h3>"]

    def attrs(r):
        a = f'data-codec="{codec}" data-n="{r["n"]}" data-rate-hz="{r["rate_hz"]:.9g}"'
        if atoms:
            a += (f' data-atom-latents="{r["atom_latents"]}" data-crossfade-ms="{r["crossfade_ms"]:.3f}"'
                  f' data-k-max="{r["k_max"]}"')
        return a

    if full:
        head = "<tr><th>n</th><th>rate (Hz)</th><th>step (s)</th>"
        head += ("<th>x</th><th>o</th><th>A (latents)</th><th>A (s)</th><th>crossfade (ms)</th><th>K max</th></tr>"
                 if atoms else "</tr>")
        rows = []
        for r in full:
            cells = f"<td>{r['n']}</td><td>{r['rate_hz']:.3f}</td><td>{r['step_seconds']:.3f}</td>"
            if atoms:
                cells += (f"<td>{r['crossfade']}</td><td>{r['overlapped']}</td><td>{r['atom_latents']}</td>"
                          f"<td>{r['atom_seconds']:.3f}</td><td>{r['crossfade_ms']:.1f}</td><td>{r['k_max']}</td>")
            cls = ' class="def"' if r["n"] == default_n else ""
            rows.append(f"<tr{cls} {attrs(r)}>{cells}</tr>")
        out.append(f'<div class="wrap"><table>{head}{"".join(rows)}</table></div>')
    if slow:
        out.append(f"<p>Rates below 1 Hz (n = {slow[0]['n']} to {slow[-1]['n']}), one cell per rate: "
                   f"{'n, rate in Hz, A in latents' if atoms else 'n, rate in Hz'}.</p>")
        cells = []
        for r in slow:
            label = f"{r['n']} &middot; {r['rate_hz']:.3f}" + (f" &middot; {r['atom_latents']}" if atoms else "")
            cells.append(f'<span class="cell" {attrs(r)}>{label}</span>')
        out.append(f'<div class="wrap"><div class="grid">{"".join(cells)}</div></div>')
    return "\n".join(out)


def build():
    sp = CODECS["encodec48"]
    parts = [
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>SCAPES generation modes</title>'
        f"<style>{CSS}</style></head><body>",
        "<h1>SCAPES generation: autoregressive and atoms</h1>",
        "<p>Three keys set the generation geometry: <code>codec</code>, <code>conditioning_rate</code> (Hz) and "
        "<code>conditioning_memory_seconds</code>. The default rate is "
        f"{DEFAULT_RATE:g} Hz for every codec. The requested rate is rounded to the nearest available rate "
        "(ties go to the smaller n), and a startup line prints the resolved geometry and one warning line per "
        "adjustment. This page is generated from the resolver, so it cannot disagree with the code.</p>",
        "<h2>Codecs</h2>", codec_summary(),
        "<h2>Autoregressive</h2>", svg_autoregressive(),
        "<p>Causal codecs generate one block of n latents per step and attend to the last W latents. "
        f"W = min(m, {CAP}), raised to n if the memory is shorter than a block.</p>",
        "<h2>Atoms</h2>", svg_atoms(),
        "<p>Non-causal codecs generate one atom per step. Its first o latents are zeroed, the next x are crossfaded with "
        "the previous atom, and the last n are new. The buffer holds K past atoms, each shifted by n.</p>",
        "<h2>Formulas</h2>",
        '<div class="formulas">'
        "x = min(&lceil;n/2&rceil;, x<sub>def</sub>) <br>"
        "o = max(n, A<sub>min</sub> &minus; x &minus; n) <br>"
        "A = o + x + n <br>"
        "m = &lceil;latent rate &times; memory seconds&rceil; latents; K = smallest K &ge; 1 with m &le; x + K&middot;n <br>"
        "n = the n between n<sub>min</sub> and n<sub>max</sub> that minimises |latent rate / n &minus; requested rate| <br>"
        f"cap: K&middot;A &le; {CAP}, else K = K<sub>max</sub> = &lfloor;{CAP}/A&rfloor;; W &le; {CAP}"
        "</div>",
        f"<p>The cap is {CAP} attended latents for every codec. A rate faster than n<sub>min</sub> or slower than "
        "n<sub>max</sub> is clipped to the nearest end; any other request is snapped. n<sub>max</sub> is the largest n "
        f"with A &le; {CAP} for atoms and {CAP} for autoregressive.</p>",
        "<h2>Worked examples</h2>", examples(),
        f"<h2>Defaults at {DEFAULT_RATE:g} Hz</h2>", defaults_table(),
        "<h2>The window</h2>",
        "<p>The window is one replaceable function, <code>crossfade_window</code> in <code>SCAPES/data/geometry.py</code>: "
        "zeros, a rise of equal power (sine), ones, and the matching fall (cosine), so rise&sup2; + fall&sup2; = 1. "
        "Changing the window means editing only that function.</p>",
        "<h2>Every conditioning rate</h2>",
        "<p>Every n from n<sub>min</sub> to n<sub>max</sub> per codec. The default row is highlighted.</p>",
    ]
    parts += [codec_table(c) for c in USER]
    parts.append("</body></html>")
    return "\n".join(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/generation_modes.html")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(build())
    print("wrote", a.out)
