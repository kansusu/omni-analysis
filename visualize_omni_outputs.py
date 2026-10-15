"""
Turn the output folder of run_qwen3_omni_multi.py into ONE self-contained HTML report
that people can read and discuss. No dependencies beyond the Python standard library.

What it does
  * python outputs -> a swim-lane timeline (one lane per participant) + a readable event table.
                      Simulation scripts are read from the <name>.events.json their run produced
                      (the script itself is never executed here), plus the printed replay
                      (<name>.sim.txt). Older dict-style python outputs are parsed directly.
  * amr outputs    -> each AMR graph drawn as a tree diagram (captions with time + sentence)
  * svg outputs    -> the model's own drawing, shown only if it actually drew something
  * text outputs   -> rendered transcript / per-person behavior notes ("Individual" tab)
  * group outputs  -> group-level simulation (<name>.group.py / .group.events.json): joint
                      attention, mutual gaze, joint/mutual gestures, joint actions ("Group" tab),
                      next to the group events derived from the individual simulation
  * every output is health-checked and gets a plain-language reason when it is broken
    (cut off, runaway repetition, bad quoting, timestamps outside the clip, ...)
  * model output that is broken is salvaged where possible instead of being thrown away

The report never trusts the model's markup: everything is HTML-escaped and the model's SVG
is shown inside an <img>, so scripts in it cannot run.

Usage:
    python visualize_omni_outputs.py omni_out                # -> omni_out/report.html
    python visualize_omni_outputs.py omni_out -o talk.html --chunk-len 80
"""
import argparse
import ast
import base64
import collections
import glob
import html
import json
import os
import re

MODALITY_COLOR = {
    "speech": "#3b82f6", "gesture": "#f59e0b", "gaze": "#14b8a6",
    "posture": "#a855f7", "other": "#94a3b8",
}
E = html.escape


# ============================ loading ============================

def load_units(d):
    """[{key, offset, files{fmt: path}, summary{fmt: check}}] sorted by offset."""
    exts = {"txt": "text", "py": "python", "amr": "amr", "svg": "svg"}
    units = {}
    for path in glob.glob(os.path.join(d, "*")):
        base = os.path.basename(path)
        side = {".group.events.json": "group_events", ".group.sim.txt": "group_sim",
                ".group.py": "group", ".group.txt": "group_text",
                ".events.json": "sim_events", ".sim.txt": "sim_txt"}
        extra = next((k for k in side if base.endswith(k)), None)  # longest suffixes first
        if extra:  # side files belong to the unit of the same stem
            stem = base[: -len(extra)]
            u = units.setdefault(stem, {"key": stem, "files": {}, "checks": {}, "offset": 0.0})
            u["files"][side[extra]] = path
            m = re.search(r"start(\d+)", stem)
            if m:
                u["offset"] = float(m.group(1))
            continue
        stem, ext = os.path.splitext(base)
        if ext[1:] in exts and "_chunk_" in stem or (ext[1:] in exts and re.search(r"start\d+", stem)):
            u = units.setdefault(stem, {"key": stem, "files": {}, "checks": {}, "offset": 0.0})
            u["files"][exts[ext[1:]]] = path
            m = re.search(r"start(\d+)", stem)
            if m:
                u["offset"] = float(m.group(1))
    sp = os.path.join(d, "summary.json")
    src = None
    if os.path.exists(sp):
        try:
            s = json.load(open(sp))
            src = s.get("input_name") or s.get("input")
            for su in s.get("units", []):
                for fmt, o in su.get("outputs", {}).items():
                    stem = os.path.splitext(os.path.basename(o["file"]))[0]
                    if stem.endswith(".group"):
                        stem = stem[: -len(".group")]
                    if stem in units:
                        units[stem]["checks"][fmt] = o.get("check", "")
                        units[stem]["offset"] = su.get("offset_s", units[stem]["offset"])
        except Exception:
            pass
    return sorted(units.values(), key=lambda u: (u["offset"], u["key"])), src


def read(path):
    with open(path, errors="replace") as f:
        return f.read()


def strip_header(t):
    """Drop the source header lines added by the runner."""
    lines = t.splitlines()
    out = []
    for ln in lines:
        if ln.startswith(("# source:", "# ::source", "SOURCE_FILE", "UNIT_FILE", "OFFSET_S", "<!-- source:")) \
                or re.match(r"\[source:.*\]$", ln):
            continue
        out.append(ln)
    return "\n".join(out).strip("\n")


def strip_fence(t):
    m = re.search(r"```[A-Za-z]*[ \t]*\n(.*?)```", t, re.S)
    closed = bool(m)
    if not m:
        m = re.search(r"```[A-Za-z]*[ \t]*\n(.*)", t, re.S)
    return (m.group(1) if m else t), (closed or "```" not in t)


# ============================ health result ============================

class Health:
    def __init__(self):
        self.level = "ok"      # ok | warn | fail
        self.notes = []

    def add(self, level, msg):
        order = {"ok": 0, "warn": 1, "fail": 2}
        if order[level] > order[self.level]:
            self.level = level
        self.notes.append((level, msg))


def repetition_ratio(items):
    items = [re.sub(r"\W+", " ", i.lower()).strip() for i in items if i.strip()]
    return len(set(items)) / len(items) if items else 1.0


# ============================ python events ============================

def split_dicts(body):
    """Top-level {...} blocks (brace matching, ignoring braces inside quotes)."""
    out, depth, start, q, esc = [], 0, None, None, False
    n = len(body)
    for i, ch in enumerate(body):
        if ch == "'" and 0 < i < n - 1 and body[i - 1].isalpha() and body[i + 1].isalpha():
            continue  # apostrophe inside a word ("That's"), never a quote boundary
        if q:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == q:
                q = None
            continue
        if ch in "'\"":
            q = ch
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                out.append(body[start:i + 1])
    return out


def loose_dict(s):
    """Fallback for dicts with broken quoting (e.g. 'That's good')."""
    d = {}
    for k in ("start", "end"):
        m = re.search(r"['\"]%s['\"]\s*:\s*(-?[\d.]+)" % k, s)
        if m:
            d[k] = float(m.group(1))
    for k in ("participant", "modality", "id"):
        m = re.search(r"['\"]%s['\"]\s*:\s*['\"]([^'\"]*)['\"]" % k, s)
        if m:
            d[k] = m.group(1)
    m = re.search(r"['\"](?:content|description)['\"]\s*:\s*(.*?)(?:,\s*['\"]target['\"]|\s*\}\s*$)", s, re.S)
    if m:
        d["content" if "content" in s else "description"] = m.group(1).strip().strip("'\"")
    m = re.search(r"['\"]target['\"]\s*:\s*(?:['\"]([^'\"]*)['\"]|(None))", s)
    if m:
        d["target"] = m.group(1)
    return d


def parse_python(text):
    body, closed = strip_fence(strip_header(text))
    info = {"participants": [], "events": [], "loose": 0, "syntax_ok": False, "closed": closed}
    try:
        ast.parse(body)
        info["syntax_ok"] = True
    except SyntaxError as e:
        info["syntax_err"] = f"{e.msg} (line {e.lineno})"
    for blk in split_dicts(body):
        try:
            d = ast.literal_eval(blk)
        except Exception:
            d = loose_dict(blk)
            if not d:
                continue
            info["loose"] += 1
        if not isinstance(d, dict):
            continue
        if "start" in d and "end" in d:
            info["events"].append(d)
        elif "id" in d:
            info["participants"].append(d)
    return info


def python_health(info, chunk_len):
    h = Health()
    ev = info["events"]
    if info["syntax_ok"]:
        h.add("ok", f"simulation script ran and produced {len(info['events'])} events" if info.get("sim")
              else "valid Python; loads without errors")
    else:
        why = info.get("syntax_err", "")
        if not info["closed"]:
            h.add("fail", f"file is cut off before the end (generation hit the token limit): {why}")
        elif info["loose"]:
            h.add("fail", f"{info['loose']} event(s) contain a quote inside a quoted string "
                          f"(e.g. 'That's good'), so Python cannot parse the file: {why}")
        else:
            h.add("fail", f"not valid Python: {why}")
        if ev:
            h.add("warn", f"salvaged {len(ev)} event(s) from the broken file")
    if ev:
        starts = [e.get("start", 0) for e in ev]
        ends = [e.get("end", 0) for e in ev]
        if len(ev) >= 5 and len(set(starts)) <= 2:
            h.add("fail", f"timestamps are degenerate: all {len(ev)} events start at "
                          f"{starts[0]:g}s (the model copied one time range instead of timing events)")
        elif max(ends) > chunk_len + 5:
            h.add("warn", f"events run to {max(ends):g}s but the clip is only {chunk_len:g}s long")
        elif max(ends) < chunk_len * 0.5:
            h.add("warn", f"events only cover the first {max(ends):g}s of a {chunk_len:g}s clip")
        rr = repetition_ratio([str(e.get("content", "")) for e in ev])
        if len(ev) > 8 and rr < 0.6:
            h.add("warn", f"{int((1 - rr) * 100)}% of event texts are repeats")
    else:
        h.add("fail", "no events could be recovered")
    return h


# ============================ AMR ============================

TOK = re.compile(r'\s*(\(|\)|"(?:[^"\\]|\\.)*"|:[^\s()]+|/|[^\s()/"]+)')


class ANode:
    def __init__(self, var=None, concept=None):
        self.var, self.concept, self.kids = var, concept, []   # kids: (role, ANode|str)


def tokenize_amr(s):
    return [m.group(1) for m in TOK.finditer(s)]


def parse_amr_graph(s):
    """Tolerant PENMAN reader. Returns (root, info) where info counts repairs."""
    toks = tokenize_amr(s)
    pos = [0]
    info = {"nodes": 0, "missing_close": 0, "stray_close": 0, "no_concept": 0, "dup_var": 0, "no_var": 0}
    seen = set()

    def node():
        pos[0] += 1  # consume "("
        n = ANode()
        info["nodes"] += 1
        # head: [var /] concept   (or nothing, or a role directly)
        t = toks[pos[0]] if pos[0] < len(toks) else None
        if t and not t.startswith(":") and t not in ("(", ")", "/"):
            if pos[0] + 1 < len(toks) and toks[pos[0] + 1] == "/":
                n.var = t
                pos[0] += 2
                if n.var in seen:
                    info["dup_var"] += 1
                seen.add(n.var)
                t2 = toks[pos[0]] if pos[0] < len(toks) else None
                if t2 and not t2.startswith(":") and t2 not in ("(", ")"):
                    n.concept = t2
                    pos[0] += 1
            else:
                n.concept = t       # "(color green)" style: treat first symbol as concept
                pos[0] += 1
        if n.concept is None:
            info["no_concept"] += 1
        if n.var is None:
            info["no_var"] += 1
        while pos[0] < len(toks):
            t = toks[pos[0]]
            if t == ")":
                pos[0] += 1
                return n
            if t.startswith(":"):
                pos[0] += 1
                if pos[0] >= len(toks):
                    break
                c = toks[pos[0]]
                if c == "(":
                    n.kids.append((t, node()))
                elif c == ")" or c.startswith(":"):
                    n.kids.append((t, "?"))
                else:
                    n.kids.append((t, c))
                    pos[0] += 1
            elif t == "(":
                n.kids.append((":?", node()))
            else:  # stray atom after concept, e.g. "(color green)" second word
                n.kids.append((":?", t))
                pos[0] += 1
        info["missing_close"] += 1
        return n

    if not toks or toks[0] != "(":
        return None, info
    root = node()
    info["stray_close"] = sum(1 for t in toks[pos[0]:] if t == ")")
    return root, info


def split_amr(text):
    """[{time, snt, src}] one entry per graph or per caption-only comment."""
    items, cur = [], {}
    buf = []

    def flush():
        nonlocal buf
        if buf:
            items.append({**cur, "src": "\n".join(buf)})
            buf = []

    for ln in strip_header(text).splitlines():
        if ln.startswith("#"):
            if buf:
                flush()
                cur = {}
            m = re.search(r"::time\s+(\S+)", ln)
            if m:
                if cur.get("time") or cur.get("snt"):
                    items.append(dict(cur))   # a caption without a graph
                    cur = {}
                cur["time"] = m.group(1)
            m = re.search(r"::snt\s+(.*)", ln)
            if m:
                cur["snt"] = m.group(1).strip()
        elif ln.strip() == "":
            if buf:
                flush()
                cur = {}
        else:
            buf.append(ln)
    flush()
    if cur.get("time") or cur.get("snt"):
        items.append(dict(cur))
    return items


def time_to_s(t):
    m = re.match(r"(\d+):(\d+)-(\d+):(\d+)", t or "")
    return (int(m[1]) * 60 + int(m[2]), int(m[3]) * 60 + int(m[4])) if m else None


def amr_health(items, parsed, chunk_len):
    h = Health()
    graphs = [p for p in parsed if p and p[0] is not None]
    caps = [i for i in items if i.get("snt")]
    if not graphs:
        h.add("fail", "no AMR graph in the output" + (f" - only {len(caps)} caption comments" if caps else ""))
    for it, p in zip(items, parsed):
        if not p or p[0] is None:
            continue
        info = p[1]
        bad = []
        if info["missing_close"]:
            bad.append(f"{info['missing_close']} unclosed bracket(s)")
        if info["stray_close"]:
            bad.append(f"{info['stray_close']} extra ')'")
        if info["no_concept"]:
            bad.append(f"{info['no_concept']} node(s) without a concept")
        if info["no_var"] and not info["no_concept"]:
            bad.append(f"{info['no_var']} node(s) lack the 'variable / concept' form, so a strict AMR parser rejects it")
        if info["dup_var"]:
            bad.append(f"variable name reused {info['dup_var']}x")
        if info["nodes"] > 40:
            concepts = collections.Counter(c for c in iter_concepts(p[0]))
            if len(concepts) < 8:
                bad.append(f"runaway repetition ({info['nodes']} nodes built from "
                           f"{len(concepts)} concepts)")
        if bad:
            h.add("fail", "invalid AMR: " + "; ".join(bad))
            break
    sn = [i.get("snt", "") for i in items if i.get("snt")]
    if len(sn) >= 6 and repetition_ratio(sn) < 0.5:
        h.add("warn", f"only {len(set(sn))} distinct sentences out of {len(sn)} (model looped)")
    spans = [time_to_s(i.get("time")) for i in items if i.get("time")]
    spans = [s for s in spans if s]
    if spans and max(e for _, e in spans) > chunk_len + 5:
        h.add("warn", f"captions run to {max(e for _, e in spans)}s but the clip is only {chunk_len:g}s long")
    if not items:
        h.add("fail", "empty output")
    if h.level == "ok":
        h.add("ok", f"{len(graphs)} graph(s) parsed")
    return h


def iter_concepts(n):
    stack = [n]
    while stack:
        x = stack.pop()
        yield x.concept or "?"
        stack.extend(c for _, c in x.kids if isinstance(c, ANode))


# ============================ svg ============================

DRAWABLE = ("<rect", "<circle", "<ellipse", "<line", "<path", "<polygon", "<polyline", "<text", "<image")


def repair_svg(svg):
    if "</svg>" in svg:
        return svg, False
    svg = svg[: svg.rfind(">") + 1]
    stack = []
    for m in re.finditer(r"<(/?)([A-Za-z][\w:.-]*)[^>]*?(/?)>", svg):
        closing, name, selfc = m.groups()
        if selfc:
            continue
        if closing:
            if name in stack:
                while stack and stack.pop() != name:
                    pass
        else:
            stack.append(name)
    return svg + "".join(f"</{n}>" for n in reversed(stack)), True


def svg_info(text):
    body, _ = strip_fence(strip_header(text))
    i = body.find("<svg")
    body = body[i:] if i >= 0 else body
    j = body.rfind("</svg>")
    closed = j != -1
    if closed:
        body = body[: j + 6]
    fixed, _ = repair_svg(body)
    # count distinct visible shapes: the model sometimes repeats one invisible full-page <path opacity=".01"/>
    seen = set()
    for ln in fixed.splitlines():
        ln = ln.strip()
        if any(t in ln for t in DRAWABLE) and not re.search(r'opacity="?0?\.0\d?"', ln):
            seen.add(ln)
    drawn = len(seen)
    css = len(re.findall(r"\.[\w-]+\s*\{", body))  # count before repair (repair drops text after the last '>')
    return {"svg": fixed, "closed": closed, "drawn": drawn, "css": css}


def svg_health(info):
    h = Health()
    if not info["closed"]:
        h.add("fail", "drawing is cut off before </svg> (token limit)")
    if info["drawn"] == 0:
        h.add("fail", f"nothing was drawn: output is only {info['css']} CSS style rules "
                      f"(the model looped on <style> and never reached the shapes)" if info["css"] > 10
              else "nothing visible was drawn (only repeated invisible full-page paths or style rules)")
    elif info["closed"]:
        h.add("ok", f"{info['drawn']} shapes/labels")
    else:
        h.add("warn", f"{info['drawn']} shapes/labels survive; the rest is missing")
    return h


# ============================ text ============================

def text_health(t):
    h = Health()
    body = strip_header(t).rstrip()
    if body and body[-1] not in ".!?)\"'*`":
        h.add("fail", "cut off mid-sentence (hit the token limit); the end of the clip is missing")
    lines = [re.sub(r"\d+:\d+|[*\-]|speaker \d+|man in \w+ shirt", " ", ln.lower()) for ln in body.splitlines()]
    lines = [ln for ln in lines if len(ln.split()) > 4]
    if len(lines) > 8 and repetition_ratio(lines) < 0.7:
        h.add("warn", "many near-identical lines (the model started repeating itself)")
    if h.level == "ok":
        h.add("ok", "complete")
    return h


def md_lite(t):
    out, in_ul = [], False
    for ln in strip_header(t).splitlines():
        s = E(ln)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        if re.match(r"\s*#{1,4} ", ln):
            if in_ul:
                out.append("</ul>")
                in_ul = False
            head = re.sub(r"^\s*#+ ", "", s)
            out.append(f"<h4>{head}</h4>")
        elif re.match(r"\s*[-*] ", ln):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append("<li>" + re.sub(r"^\s*[-*] ", "", s) + "</li>")
        elif ln.strip():
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append(f"<p>{s}</p>")
    if in_ul:
        out.append("</ul>")
    return "\n".join(out)


# ============================ drawing: timeline ============================

def timeline_svg(participants, events, t0, t1, width=1040):
    pid = {}
    for p in participants:
        pid[str(p.get("id"))] = p.get("description", "")
    lanes = []
    for e in events:
        k = str(e.get("participant", "?"))
        if k not in lanes:
            lanes.append(k)
    lanes = sorted(lanes, key=lambda k: ("+" in k, not k.startswith("p"), k))
    left, top, sub_h, pad = 190, 34, 22, 10
    span = max(t1 - t0, 1e-6)
    sx = lambda t: left + (t - t0) / span * (width - left - 20)

    rows, y = {}, top
    placed = {}
    for lane in lanes:
        subs = []   # end-x of last event in each sub-row
        for e in sorted((x for x in events if str(x.get("participant", "?")) == lane),
                        key=lambda x: x.get("start", 0)):
            a, b = sx(e["start"] + 0), max(sx(e["end"]), sx(e["start"]) + 4)
            for i, last in enumerate(subs):
                if a >= last + 2:
                    subs[i] = b
                    placed[id(e)] = i
                    break
            else:
                subs.append(b)
                placed[id(e)] = len(subs) - 1
        rows[lane] = (y, max(len(subs), 1))
        y += max(len(subs), 1) * sub_h + pad
    H = y + 6
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {H}" class="tl" role="img">']
    step = 10 if span <= 150 else 30
    t = (int(t0) // step) * step
    while t <= t1 + 1e-6:
        if t >= t0:
            o.append(f'<line x1="{sx(t):.1f}" y1="{top - 6}" x2="{sx(t):.1f}" y2="{H - 4}" class="grid"/>'
                     f'<text x="{sx(t):.1f}" y="{top - 12}" class="tick" text-anchor="middle">{int(t // 60)}:{int(t % 60):02d}</text>')
        t += step
    for lane in lanes:
        ly, n = rows[lane]
        desc = pid.get(lane, "")
        o.append(f'<rect x="0" y="{ly - 3}" width="{width}" height="{n * sub_h + 2}" class="lane"/>')
        o.append(f'<text x="8" y="{ly + 12}" class="lanelabel"><title>{E(lane + ": " + desc)}</title>'
                 f'{E(lane)} {E(desc[:24] + ("…" if len(desc) > 24 else ""))}</text>')
    for e in events:
        lane = str(e.get("participant", "?"))
        ly, _ = rows[lane]
        yy = ly + placed.get(id(e), 0) * sub_h
        a = sx(e["start"])
        w = max(sx(e["end"]) - a, 4)
        col = MODALITY_COLOR.get(str(e.get("modality", "other")), MODALITY_COLOR["other"])
        txt = str(e.get("content", ""))
        tip = f'{e["start"]:g}–{e["end"]:g}s · {e.get("modality", "")}\n{txt}'
        o.append(f'<g><title>{E(tip)}</title><rect x="{a:.1f}" y="{yy}" width="{w:.1f}" height="{sub_h - 4}" '
                 f'rx="3" fill="{col}" opacity=".85"/>')
        fit = int((w - 6) / 6.2)
        if fit >= 4:
            o.append(f'<text x="{a + 4:.1f}" y="{yy + 12}" class="evt">{E(txt[:fit] + ("…" if len(txt) > fit else ""))}</text>')
        o.append("</g>")
    o.append("</svg>")
    return "\n".join(o)


def legend():
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{c}"></i>{k}</span>' for k, c in MODALITY_COLOR.items()) + "</div>"


GROUP_NAMES = {"JointAttention": "joint attention", "MutualGaze": "look at each other",
               "JointGesture": "joint gesture", "MutualGesture": "gesture at each other",
               "JointAction": "joint action"}


def label_derived(events):
    """Readable content for derived group events, e.g. 'joint attention → blocks (derived)'."""
    out = []
    for e in events:
        name = GROUP_NAMES.get(e.get("type"), e.get("type", "group event"))
        tgt = f" → {e['target']}" if e.get("target") else ""
        out.append({**e, "content": f"{name}{tgt} (derived)"})
    return out


def events_table(events, offset, open_=False):
    rows = []
    for e in sorted(events, key=lambda x: (x.get("start", 0), str(x.get("participant")))):
        col = MODALITY_COLOR.get(str(e.get("modality", "other")), MODALITY_COLOR["other"])
        a = offset + e.get("start", 0)
        kind = GROUP_NAMES.get(e.get("type"), e.get("type", "")) if e.get("group") else e.get("modality", "")
        what = str(e.get("content", ""))
        rows.append(f'<tr><td class="num">{int(a // 60)}:{int(a % 60):02d}</td><td>{E(str(e.get("participant", "")))}</td>'
                    f'<td><i class="dot" style="background:{col}"></i>{E(str(kind))}</td>'
                    f'<td>{E(what)}</td><td>{E(str(e.get("target") or ""))}</td></tr>')
    return ('<details' + (' open' if open_ else '') + '><summary>All ' + str(len(rows)) +
            ' events as a table (times are in the full video)</summary>'
            '<table class="ev"><tr><th>time</th><th>who</th><th>kind</th><th>what</th><th>to</th></tr>'
            + "".join(rows) + "</table></details>")


# ============================ drawing: AMR tree ============================

MAX_DEPTH, MAX_REPEAT = 7, 2


def build_tree(n, path=(), depth=0, budget=None):
    """-> dict(label, kids[(role, tree)], kind). Cuts loops/depth, never raises."""
    budget = budget if budget is not None else [60]
    label = (n.var + " / " if n.var else "") + (n.concept or "?")
    concept = n.concept or "?"
    if depth >= MAX_DEPTH or path.count(concept) >= MAX_REPEAT or budget[0] <= 0:
        return {"label": "…", "kids": [], "kind": "cut"}
    budget[0] -= 1
    t = {"label": label, "kids": [], "kind": "node" if n.concept else "bad", "concept": concept}
    for role, c in n.kids:
        if isinstance(c, ANode):
            t["kids"].append((role, build_tree(c, path + (concept,), depth + 1, budget)))
        else:
            t["kids"].append((role, {"label": c, "kids": [], "kind": "leaf"}))
    return t


def amr_svg(root):
    tree = build_tree(root)
    cols, ys, counter = {}, {}, [0]

    def width(lbl):
        return max(34, len(lbl) * 7 + 16)

    def measure(t, d):
        cols[d] = max(cols.get(d, 0), width(t["label"]))
        for _, c in t["kids"]:
            measure(c, d + 1)
    measure(tree, 0)
    xs, x = {}, 12
    for d in sorted(cols):
        xs[d] = x
        x += cols[d] + 88
    W = x - 70
    row = 30

    def place(t, d):
        if not t["kids"]:
            t["y"] = counter[0] * row + 18
            counter[0] += 1
        else:
            for _, c in t["kids"]:
                place(c, d + 1)
            t["y"] = sum(c["y"] for _, c in t["kids"]) / len(t["kids"])
        t["d"] = d
    place(tree, 0)
    H = max(counter[0], 1) * row + 14
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {max(W, 200)} {H}" class="amr" '
         f'style="max-width:{max(W, 200)}px" role="img">']

    def draw(t):
        x0 = xs[t["d"]]
        w = width(t["label"])
        for role, c in t["kids"]:
            cx = xs[c["d"]]
            mx = x0 + w + 14
            o.append(f'<path d="M{x0 + w} {t["y"]:.1f} C{mx} {t["y"]:.1f} {mx} {c["y"]:.1f} {cx} {c["y"]:.1f}" class="edge"/>')
            o.append(f'<text x="{(x0 + w + cx) / 2:.1f}" y="{(t["y"] + c["y"]) / 2 - 3:.1f}" class="role" text-anchor="middle">{E(role)}</text>')
            draw(c)
        cls = {"node": "n-node", "bad": "n-bad", "leaf": "n-leaf", "cut": "n-cut"}[t["kind"]]
        o.append(f'<rect x="{x0}" y="{t["y"] - 11:.1f}" width="{w}" height="22" rx="{11 if t["kind"] == "leaf" else 5}" class="{cls}"/>'
                 f'<text x="{x0 + w / 2}" y="{t["y"] + 4:.1f}" text-anchor="middle" class="nlabel">{E(t["label"])}</text>')
    draw(tree)
    o.append("</svg>")
    return "\n".join(o)


# ============================ page assembly ============================

BADGE = {"ok": ("✓", "ok"), "warn": ("!", "warn"), "fail": ("✕", "fail")}


def notes_html(h):
    return '<ul class="notes">' + "".join(
        f'<li class="{lv}"><b>{BADGE[lv][0]}</b> {E(m)}</li>' for lv, m in h.notes) + "</ul>"


def data_img(svg):
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def mm(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


SIM_KEYS = {"python": ("python", "sim_events", "sim_txt"),
            "group": ("group", "group_events", "group_sim")}


def sim_section(u, kind, chunk_len, off):
    """Panel for a simulation output. Returns (html, Health, info, derived_group_events)."""
    code_k, ev_k, txt_k = SIM_KEYS[kind]
    info, derived = None, []
    if ev_k in u["files"]:
        try:
            d = json.load(open(u["files"][ev_k]))
            info = {"participants": d.get("participants", []), "events": d.get("events", []),
                    "loose": 0, "syntax_ok": True, "closed": True, "sim": True}
            derived = label_derived(d.get("derived_group_events", []))
            chunk_len = d.get("clip_s") or chunk_len
        except Exception:
            info = None
    if info is None:
        info = parse_python(read(u["files"][code_k])) if code_k in u["files"] else \
            {"participants": [], "events": [], "loose": 0, "syntax_ok": False, "closed": True}
    if kind == "group" and info.get("sim") and not info["events"]:
        h = Health()
        h.add("ok", "script ran; the model reported no group-level events in this clip")
    else:
        h = python_health(info, chunk_len)
    chk = u["checks"].get(kind, "")
    if chk.startswith("SALVAGED"):
        h.add("warn", chk.split(";")[0] + " before running")
    if "RUN ERROR" in chk:
        h.add("fail", "the simulation script crashed: " + chk.split("RUN ERROR:")[-1].strip())
    body = notes_html(h)
    shown = info["events"] + (derived if kind == "python" else [])
    if shown:
        t1 = max(chunk_len, max(e["end"] for e in shown))
        if kind == "python" and derived:
            body += ('<p class="sub">Lanes like <b>p1+p3</b> are group-level events derived from '
                     'these individual events.</p>')
        body += legend()
        body += '<div class="scroll">' + timeline_svg(info["participants"], shown, 0, t1) + "</div>"
        body += events_table(shown, off)
    if txt_k in u["files"]:
        body += ('<details open><summary>What running the simulation prints</summary><pre>'
                 + E(read(u["files"][txt_k])[:20000]) + "</pre></details>")
    return body, h, info, derived


def build_unit(u, chunk_len):
    off, res, panels, health = u["offset"], {}, {}, {}
    # ---- text
    if "text" in u["files"]:
        t = read(u["files"]["text"])
        h = text_health(t)
        health["text"] = h
        panels["text"] = notes_html(h) + '<div class="prose">' + md_lite(t) + "</div>"
    if "group_text" in u["files"]:
        t = read(u["files"]["group_text"])
        h = text_health(t)
        health["group_text"] = h
        panels["group_text"] = notes_html(h) + '<pre class="grp">' + E(strip_header(t)) + "</pre>"
    # ---- python (individual simulation) and group (group-level simulation)
    py_events, derived = [], []
    if "python" in u["files"] or "sim_events" in u["files"]:
        body, h, info, d = sim_section(u, "python", chunk_len, off)
        health["python"], panels["python"] = h, body
        derived = d
        if not any("degenerate" in m for _, m in h.notes):
            py_events = [{**e, "start": e["start"] + off, "end": e["end"] + off}
                         for e in info["events"] + derived]
        res["py_participants"] = info["participants"]
    if "group" in u["files"] or "group_events" in u["files"]:
        body, h, info, _ = sim_section(u, "group", chunk_len, off)
        if derived:
            body += ('<div class="card"><div class="cap"><b>For comparison:</b> group events derived '
                     'from the individual simulation (overlapping gaze/gestures with the same '
                     'target, or aimed at each other)</div>' + events_table(derived, off, open_=True)
                     + "</div>")
        health["group"], panels["group"] = h, body
    # ---- amr
    if "amr" in u["files"]:
        items = split_amr(read(u["files"]["amr"]))
        parsed = [parse_amr_graph(i["src"]) if i.get("src") else None for i in items]
        h = amr_health(items, parsed, chunk_len)
        health["amr"] = h
        body = notes_html(h)
        caps = collections.OrderedDict()
        cards = []
        for it, p in zip(items, parsed):
            cap = f'<div class="cap"><b>{E(it.get("time", ""))}</b> {E(it.get("snt", ""))}</div>' if (it.get("time") or it.get("snt")) else ""
            if p and p[0] is not None:
                cards.append(f'<div class="card">{cap}<div class="scroll">{amr_svg(p[0])}</div>'
                             f'<details><summary>PENMAN source</summary><pre>{E(it["src"][:3000])}</pre></details></div>')
            elif it.get("snt"):
                k = it["snt"]
                caps.setdefault(k, []).append(it.get("time", ""))
            elif it.get("src"):
                cards.append(f'<div class="card"><div class="cap">not a parsable graph</div><pre>{E(it["src"][:600])}</pre></div>')
        if caps:
            body += ('<div class="card"><div class="cap"><b>Captions the model wrote (no graph attached)</b>'
                     ' - repeats are merged</div><table class="ev"><tr><th>times</th><th>sentence</th></tr>')
            for s, times in caps.items():
                body += (f'<tr><td class="num">{E(times[0])}{" ×" + str(len(times)) if len(times) > 1 else ""}</td>'
                         f'<td>{E(s)}</td></tr>')
            body += "</table></div>"
        panels["amr"] = body + "".join(cards)
    # ---- svg
    if "svg" in u["files"]:
        info = svg_info(read(u["files"]["svg"]))
        h = svg_health(info)
        health["svg"] = h
        body = notes_html(h)
        if info["drawn"]:
            body += (f'<div class="card"><div class="cap">Model-drawn diagram (as generated'
                     f'{"" if info["closed"] else ", missing tags closed"})</div>'
                     f'<img class="msvg" alt="model SVG" src="{data_img(info["svg"])}"></div>')
        panels["svg"] = body
    return {"u": u, "panels": panels, "health": health, "events": py_events,
            "participants": res.get("py_participants", [])}


CSS = """
:root{--bg:#f7f7f5;--fg:#1c1c1a;--mut:#6b6b66;--card:#fff;--line:#e2e2dc;--ok:#15803d;--warn:#b45309;--fail:#b91c1c;
--okbg:#e7f5ec;--warnbg:#fdf3e1;--failbg:#fcebea;--acc:#2563eb}
@media(prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--mut:#9a9a94;--card:#1f1f1d;--line:#34342f;--ok:#4ade80;
--warn:#fbbf24;--fail:#f87171;--okbg:#17301f;--warnbg:#33290f;--failbg:#3a1a19;--acc:#60a5fa}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}h4{margin:14px 0 4px}
.sub{color:var(--mut);margin:0 0 18px}
table{border-collapse:collapse;width:100%}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-size:12px;color:var(--mut);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.num{white-space:nowrap;font-variant-numeric:tabular-nums;color:var(--mut)}
.matrix td{text-align:center}.matrix td:first-child,.matrix th:first-child{text-align:left}
.pill{display:inline-block;min-width:28px;padding:1px 8px;border-radius:99px;font-weight:700;font-size:13px;cursor:help}
.pill.ok{background:var(--okbg);color:var(--ok)}.pill.warn{background:var(--warnbg);color:var(--warn)}.pill.fail{background:var(--failbg);color:var(--fail)}
.pill.na{color:var(--mut)}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 14px}
.tabs button{border:1px solid var(--line);background:var(--card);color:var(--fg);padding:6px 12px;border-radius:8px;cursor:pointer;font:inherit}
.tabs button[aria-selected=true]{border-color:var(--acc);color:var(--acc);font-weight:600}
.chunk{display:none}.chunk.on{display:block}
.fmt{display:none}.fmt.on{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0}
.cap{margin-bottom:6px}.cap b{color:var(--acc);margin-right:6px}
.notes{list-style:none;margin:0 0 10px;padding:0}.notes li{padding:6px 10px;border-radius:8px;margin:4px 0}
.notes li.ok{background:var(--okbg);color:var(--ok)}.notes li.warn{background:var(--warnbg);color:var(--warn)}.notes li.fail{background:var(--failbg);color:var(--fail)}
.notes b{margin-right:6px}
.scroll{overflow-x:auto}svg.tl{width:100%;min-width:760px;height:auto}svg.amr{width:100%;height:auto;display:block}
.lane{fill:var(--card);stroke:var(--line)}.grid{stroke:var(--line);stroke-dasharray:3 3}.tick{fill:var(--mut);font-size:11px}
.lanelabel{fill:var(--fg);font-size:12px}.evt{fill:#fff;font-size:11px;pointer-events:none}
.edge{fill:none;stroke:var(--mut);stroke-width:1.2}.role{fill:var(--acc);font-size:11px}
.n-node{fill:var(--okbg);stroke:var(--ok)}.n-leaf{fill:var(--card);stroke:var(--mut)}.n-bad{fill:var(--warnbg);stroke:var(--warn)}
.n-cut{fill:var(--failbg);stroke:var(--fail);stroke-dasharray:3 3}.nlabel{fill:var(--fg);font-size:12px}
.legend{display:flex;gap:14px;font-size:13px;color:var(--mut);margin:4px 0 8px}.legend i,.dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px}
.ev{font-size:14px}.msvg{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}
pre{white-space:pre-wrap;word-break:break-word;font-size:12px;background:var(--bg);padding:8px;border-radius:6px;max-height:320px;overflow:auto}
.prose{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:6px 16px;max-height:640px;overflow:auto}
.prose p,.prose li{margin:3px 0}details{margin:8px 0}summary{cursor:pointer;color:var(--mut)}
"""

JS = """
function show(group,idx){document.querySelectorAll('[data-g="'+group+'"]').forEach(function(e){e.classList.toggle('on',e.dataset.i==idx)});
document.querySelectorAll('[data-t="'+group+'"] button').forEach(function(b){b.setAttribute('aria-selected',b.dataset.i==idx)})}
"""

FMT_LABEL = {"text": "Individual", "group_text": "Group (text)", "group": "Group (sim)",
             "python": "Python", "amr": "AMR", "svg": "SVG"}


def build_page(units, src, chunk_len):
    built = [build_unit(u, chunk_len) for u in units]
    fmts = [f for f in ("text", "group_text", "group", "python", "amr", "svg") if any(f in b["health"] for b in built)]
    title = f"Omni report - {src}" if src else "Omni report"
    o = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
         f'<title>{E(title)}</title><style>{CSS}</style></head><body><main>']
    o.append(f'<h1>{E(title)}</h1><p class="sub">{len(built)} chunk(s) · each output was parsed and checked; '
             f'✓ usable, ! usable with problems, ✕ broken (hover a badge for the reason)</p>')
    # matrix
    o.append('<h2>Health at a glance</h2><div class="card scroll"><table class="matrix"><tr><th>chunk</th>' +
             "".join(f"<th>{FMT_LABEL[f]}</th>" for f in fmts) + "</tr>")
    for b in built:
        o.append(f'<tr><td>{mm(b["u"]["offset"])} – {mm(b["u"]["offset"] + chunk_len)}</td>')
        for f in fmts:
            h = b["health"].get(f)
            if not h:
                o.append('<td><span class="pill na">–</span></td>')
            else:
                tip = " | ".join(m for _, m in h.notes)
                o.append(f'<td><span class="pill {h.level}" title="{E(tip)}">{BADGE[h.level][0]}</span></td>')
        o.append("</tr>")
    o.append("</table></div>")
    # whole-video timeline
    allev = [e for b in built for e in b["events"]]
    skipped = [mm(b["u"]["offset"]) for b in built if b["health"].get("python") and not b["events"]
               and "python" in b["panels"]]
    if allev:
        parts, seen = [], set()
        for b in built:
            for p in b["participants"]:
                k = (p.get("id"), p.get("description"))
                if k not in seen and not any(q.get("id") == p.get("id") for q in parts):
                    seen.add(k)
                    parts.append(p)
        o.append('<h2>Whole video: who did what, when</h2><div class="card">'
                 '<p class="sub">Python events from all chunks on one time axis. Participant ids (p1, p2…) are '
                 'assigned separately in each chunk, so the same id can be a different person in another chunk. '
                 + (f'Chunks with unusable timestamps are left out: {", ".join(skipped)}.' if skipped else "") +
                 "</p>" + legend() + '<div class="scroll">' +
                 timeline_svg(parts, allev, 0, max(e["end"] for e in allev)) + "</div></div>")
    # per-chunk
    o.append('<h2>Chunk details</h2><div class="tabs" data-t="c">' + "".join(
        f'<button data-i="{i}" aria-selected="{"true" if i == 0 else "false"}" onclick="show(\'c\',{i})">'
        f'{mm(b["u"]["offset"])}–{mm(b["u"]["offset"] + chunk_len)}</button>' for i, b in enumerate(built)) + "</div>")
    for i, b in enumerate(built):
        o.append(f'<section class="chunk{" on" if i == 0 else ""}" data-g="c" data-i="{i}">')
        have = [f for f in fmts if f in b["panels"]]
        g = f"f{i}"
        o.append(f'<div class="tabs" data-t="{g}">' + "".join(
            f'<button data-i="{j}" aria-selected="{"true" if j == 0 else "false"}" onclick="show(\'{g}\',{j})">'
            f'{FMT_LABEL[f]} <span class="pill {b["health"][f].level}">{BADGE[b["health"][f].level][0]}</span></button>'
            for j, f in enumerate(have)) + "</div>")
        for j, f in enumerate(have):
            o.append(f'<div class="fmt{" on" if j == 0 else ""}" data-g="{g}" data-i="{j}">{b["panels"][f]}</div>')
        o.append("</section>")
    o.append(f"</main><script>{JS}</script></body></html>")
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="output folder of run_qwen3_omni_multi.py (omni_out)")
    ap.add_argument("-o", "--out", help="HTML file to write (default: <dir>/report.html)")
    ap.add_argument("--chunk-len", type=float, default=80, help="clip length in seconds (default 80)")
    args = ap.parse_args()
    units, src = load_units(args.dir)
    if not units:
        raise SystemExit(f"No omni output files found in {args.dir}")
    out = args.out or os.path.join(args.dir, "report.html")
    with open(out, "w") as f:
        f.write(build_page(units, src, args.chunk_len))
    print(f"Wrote {out}  ({len(units)} chunks)")


if __name__ == "__main__":
    main()
