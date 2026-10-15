"""
Video -> Qwen3-Omni -> any combination of text / python / amr / svg outputs.

Step 1 (optional clip): if --start and/or --end is given, that span is cut out of the
input video with ffmpeg into a new file. If neither is given the original video is
used untouched. Optionally --chunk splits it into overlapping chunks (same 80s/75s
scheme as chunk_video.sh), or --video can be a directory of chunks you already made.
Step 2: Qwen3-Omni (vLLM offline API, same pattern as run_qwen3_omni.py) is run once
per unit and per requested output format, each with its own prompt. Audio is passed
as a separate 16 kHz wav (same as extract_audio.sh), not via use_audio_in_video.
Step 3: each answer is cleaned (code fences stripped), sanity-checked and saved.

The python output is a runnable SIMULATION of the clip: the model builds a class hierarchy
on omni_sim's base classes (Person/Thing entities with attributes, Utterance/Gesture/...
events) and records timed say/gesture/look/posture/act calls. Broken statements are
salvaged, unsafe statements (imports, file/system access, eval, ...) are removed, then the
script is saved and run once: its printed replay goes to <name>.sim.txt and its events to
<name>.events.json. omni_sim.py is copied into --out-dir so the scripts run standalone:
    python omni_out/<name>.py --speed 1

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_qwen3_omni_multi.py --video talk.mp4 --outputs all
    python run_qwen3_omni_multi.py --video talk.mp4 --start 00:01:30 --end 00:02:10 --outputs text,svg
    python run_qwen3_omni_multi.py --video talk.mp4 --outputs python,amr --no-audio
    python run_qwen3_omni_multi.py --video long.mp4 --chunk --outputs all
    python run_qwen3_omni_multi.py --video chunks/ --outputs text,svg   # existing chunk dir
    python run_qwen3_omni_multi.py --video talk.mp4 --chunk --outputs python,group

--outputs: comma list of text, group, python, amr, svg, or "all".
  text   = per-person transcript + behavior               -> <name>.txt
  group_text = group-level events as text                 -> <name>.group.txt
  python = simulation of individual events                -> <name>.py
  group  = simulation of group-level events only          -> <name>.group.py
           (joint attention, mutual gaze, joint/mutual gestures, joint actions)
  Every simulation run also prints group events DERIVED from its individual events
  (overlapping gaze/gestures with the same target, or aimed at each other).
--start/--end: seconds (90, 90.5) or [HH:]MM:SS. Either may be given alone.
Needs ffmpeg/ffprobe on PATH for clipping, chunking and audio extraction.
"""
import argparse
import ast
import glob
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
ALL_FORMATS = ["text", "group_text", "group", "python", "amr", "svg"]
# python = simulation of individual events (<name>.py)
# group  = simulation of group-level events only (<name>.group.py): joint attention,
#          mutual gaze, joint/mutual gestures, joint actions
EXT = {"text": "txt", "group_text": "group.txt", "group": "group.py", "python": "py",
       "amr": "amr", "svg": "svg"}

PROMPTS = {
    "text": (
        "Analyze this video clip in detail, covering both audio and visual behavior:\n"
        "1. TRANSCRIPT: Transcribe what is said, word for word, labeling who speaks "
        "each line (e.g. 'woman on the left').\n"
        "2. NON-VERBAL BEHAVIOR: Describe gestures, gaze direction, posture, and "
        "other notable non-verbal behavior for each participant.\n"
        "Give timestamps relative to the start of the clip (mm:ss) throughout."
    ),
    "group_text": (
        "Watch this video clip and describe ONLY its GROUP-LEVEL events: moments where two "
        "or more people act together or toward each other. Do not describe what one person "
        "does alone. Group-level events are:\n"
        "1. JOINT ATTENTION: 2+ people look at the same person or object at the same time.\n"
        "2. MUTUAL GAZE: people look at each other.\n"
        "3. JOINT GESTURE: 2+ people point or gesture at the same person or object.\n"
        "4. MUTUAL GESTURE: people point or gesture at each other.\n"
        "5. JOINT ACTION: 2+ people do something together (e.g. on the same object).\n"
        "Refer to people by appearance (e.g. 'man in red shirt') and always name everyone "
        "involved and the shared target. Write one event per line as "
        "'[mm:ss-mm:ss] TYPE: <participants> -> <target>: <short description>', with "
        "timestamps relative to the start of the clip, in time order. If nobody acts "
        "together during a span, say so instead of inventing an event."
    ),
    "group": (
        "Watch this video clip (it is about {CLIP} seconds long) and write a Python "
        "program that SIMULATES ONLY ITS GROUP-LEVEL EVENTS: moments where two or more "
        "people act together or toward each other. Individual behavior is recorded "
        "elsewhere, so do NOT record what one person does alone. Group-level events are:\n"
        "  1. joint attention: 2+ people look at the same person or object at the same time;\n"
        "  2. mutual gaze: people look at each other;\n"
        "  3. joint gesture: 2+ people point or gesture at the same person or object;\n"
        "  4. mutual gesture: people point or gesture at each other;\n"
        "  5. joint action: 2+ people do something together (e.g. on the same object).\n"
        "Create a well-formed class hierarchy and object structure so that the code can "
        "later be improved with additional tools (e.g. people as Person objects with "
        "attributes such as appearance, clothing color, voice type). This is only an "
        "example: you do not have to implement any external tool.\n"
        "Use this API, which is already imported (from omni_sim import *):\n"
        "  scene = Scene()\n"
        "  p1 = Person(scene, \"p1\", appearance=\"...\", clothing_color=\"blue\")   # everyone visible\n"
        "  blocks = Thing(scene, \"blocks\", description=\"pile of colored blocks\")\n"
        "  g = scene.group(p1, p3)                      # the people acting together\n"
        "  g.look(start, end, at=blocks)                # joint attention\n"
        "  g.look_at_each_other(start, end)             # mutual gaze\n"
        "  g.gesture(start, end, \"points\", at=p2)       # joint gesture at the same target\n"
        "  g.gesture_at_each_other(start, end, \"points\")\n"
        "  g.act(start, end, \"what they do together\", on=blocks)\n"
        "  scene.group(p1, p2, p3, p4).look(start, end, at=blocks)   # everyone together\n"
        "Rules: start/end are seconds from the start of this clip, between 0 and {CLIP}, "
        "timed where the group event actually happens. Each group event once, one per "
        "line, in time order. Use double quotes for strings. If nobody acts together in "
        "the clip, define the people and record no events. Do not import anything except "
        "dataclasses, typing or enum; no file, network or system access; do not call "
        "scene.run(). Output ONLY one ```python code block."
    ),
    "python": (
        "Watch this video clip (it is about {CLIP} seconds long) and write a Python "
        "program that SIMULATES it: when run, it replays who did what, when, and to whom.\n"
        "Create a well-formed class hierarchy and object structure so that the code can "
        "later be improved with additional tools. For example, a person can be modeled as "
        "a Person class and instantiated with attributes (appearance, clothing color, voice "
        "type, ...), so that the utterances a person makes could later be replaced by a TTS "
        "module. This is only an example: you do not have to implement TTS or any other "
        "external tool.\n"
        "Build on these base classes, which are already imported (from omni_sim import *):\n"
        "  Scene; Entity(scene, id, description=\"\", **attributes) with subclasses Person "
        "and Thing; Event with subclasses Utterance, Gesture, Gaze, Posture, Action.\n"
        "Subclass them where it helps (e.g. class Builder(Person) with a role, or a custom "
        "Event subclass), and give people attributes as keyword arguments. Record what "
        "happens ONLY through these Person methods, so the scene can be replayed:\n"
        "  scene = Scene()\n"
        "  p1 = Person(scene, \"p1\", appearance=\"...\", clothing_color=\"blue\", voice_type=\"...\")\n"
        "  blocks = Thing(scene, \"blocks\", description=\"pile of colored blocks\")\n"
        "  p1.say(start, end, \"exact words spoken\", to=p2)    # to=None if addressing everyone\n"
        "  p1.gesture(start, end, \"what the gesture is\", at=blocks)\n"
        "  p1.look(start, end, at=p2)\n"
        "  p1.posture(start, end, \"what the posture is\")\n"
        "  p1.act(start, end, \"what they do\", on=blocks)\n"
        "  p1.do(MyEvent(p1, start, end, \"content\", target))  # for custom Event subclasses\n"
        "Rules: start/end are seconds from the start of this clip, between 0 and {CLIP}, "
        "and are timed where each event actually happens. Event calls go one per line, in "
        "time order, after the class definitions and objects. Use double quotes for "
        "strings. Describe each distinct event once; at most about 40 events. Do not import "
        "anything except dataclasses, typing or enum; no file, network or system access; "
        "do not call scene.run(). Output ONLY one ```python code block."
    ),
    "amr": (
        "Watch this video clip and describe its main events as Abstract Meaning "
        "Representation (AMR) graphs in PENMAN notation, one graph per distinct "
        "event or utterance, in chronological order. Output ONLY a ```amr code "
        "block containing the graphs separated by blank lines. Before each graph "
        "put a comment line like '# ::time 00:03-00:07 ::snt <short sentence>'. "
        "Use PropBank frames (e.g. speak-01, look-01) and standard roles "
        "(:ARG0, :ARG1, :time, :mod)."
    ),
    "svg": (
        "Watch this video clip and draw a single SVG diagram that summarizes it: "
        "a timeline with one horizontal lane per participant, boxes for speech and "
        "non-verbal events placed by time, with short text labels. Output ONLY one "
        "```svg code block containing a complete, valid, self-contained <svg> "
        "element with xmlns and a viewBox. Keep it compact: at most about 40 "
        "shapes, no repeated boilerplate, short labels, and always finish with </svg>."
    ),
}


# ---------- step 1: optional clipping ----------

def parse_time(s):
    if s is None:
        return None
    parts = str(s).split(":")
    if len(parts) > 3:
        raise ValueError(f"bad time: {s}")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return secs


def need_ffmpeg():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise SystemExit("ffmpeg/ffprobe not found on PATH (needed for clipping / chunking / audio)")


def duration(video):
    need_ffmpeg()
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def cut(video, start, length, outfile, accurate):
    """Cut [start, start+length) into outfile. length=None -> to end of video.
    Default is stream copy (like chunk_video.sh: this ffmpeg may lack libx264), so
    cuts snap to keyframes. --accurate re-encodes for frame-exact cuts."""
    need_ffmpeg()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start or 0.0), "-i", video]
    if length is not None:
        cmd += ["-t", str(length)]
    cmd += ["-c:v", "libx264", "-c:a", "aac"] if accurate else ["-c", "copy"]
    subprocess.run(cmd + [outfile], check=True)


def maybe_clip(video, start, end, out_dir, accurate):
    """Return (path, offset_seconds): the original if no start/end given, else a new clip."""
    if start is None and end is None:
        return video, 0.0
    if start is not None and end is not None and end <= start:
        raise SystemExit(f"--end ({end}) must be greater than --start ({start})")
    base = os.path.splitext(os.path.basename(video))[0]
    tag = f"{'start' if start is None else int(start)}_{'end' if end is None else int(end)}"
    clip = os.path.join(out_dir, f"{base}_clip_{tag}.mp4")
    cut(video, start, None if end is None else end - (start or 0.0), clip, accurate)
    print(f"Clipped -> {clip}")
    return clip, start or 0.0


def make_chunks(video, offset, out_dir, chunk_len, stride, accurate):
    """Overlapping chunks like chunk_video.sh. Returns [(path, absolute_offset_s)]."""
    total = duration(video)
    cdir = os.path.join(out_dir, "chunks")
    os.makedirs(cdir, exist_ok=True)
    units, i, st = [], 0, 0
    while st < int(total):
        f = os.path.join(cdir, f"chunk_{i:03d}_start{int(offset) + st:05d}.mp4")
        cut(video, st, chunk_len, f, accurate)
        units.append((f, offset + st))
        i, st = i + 1, st + stride
    print(f"Created {len(units)} chunks in {cdir}")
    return units


def units_from_dir(d):
    files = sorted(glob.glob(os.path.join(d, "*.mp4")))
    if not files:
        raise SystemExit(f"No .mp4 files found in {d}")
    units = []
    for f in files:
        m = re.search(r"start(\d+)", os.path.basename(f))
        units.append((f, float(m.group(1)) if m else 0.0))
    return units


def extract_wav(video, out_dir):
    """Reuse or extract a 16 kHz mono wav next to the video; None if it has no audio."""
    need_ffmpeg()
    wav = os.path.splitext(video)[0] + ".wav"  # next to the video, like extract_audio.sh
    if os.path.exists(wav):
        return wav
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", wav],
        capture_output=True,
    )
    return wav if r.returncode == 0 and os.path.exists(wav) else None


# ---------- step 3: post-processing ----------

def strip_fence(text, lang):
    # any language tag: the model writes ```py, ```python, ```penman, ```xml ...
    m = re.search(r"```[A-Za-z]*[ \t]*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    # unterminated fence (hit max_tokens)
    m = re.search(r"```[A-Za-z]*[ \t]*\n(.*)", text, re.S)
    return (m.group(1) if m else text).strip()


def repair_svg(svg):
    """Close tags left open by a truncated generation. Returns (svg, was_repaired)."""
    if "</svg>" in svg:
        return svg, False
    svg = svg[: svg.rfind(">") + 1]  # drop a half-written tag/attribute
    stack = []
    for m in re.finditer(r"<(/?)([A-Za-z][\w:.-]*)[^>]*?(/?)>", svg):
        closing, name, selfclose = m.groups()
        if selfclose:
            continue
        if closing:
            if name in stack:
                while stack and stack.pop() != name:
                    pass
        else:
            stack.append(name)
    if not stack:  # already balanced (e.g. a self-closed <svg/>)
        return svg, False
    return svg + "".join(f"</{n}>" for n in reversed(stack)), True


def salvage_python(code):
    """Keep every statement that parses; drop broken ones (bad quoting, cut-off tail).
    A statement is the longest run of lines from a starting line (through its indented
    block, e.g. a whole class) that parses. Returns (code, dropped_line_count)."""
    lines, good, dropped, i = code.splitlines(), [], 0, 0

    def parses(chunk):
        try:
            ast.parse("\n".join(chunk))
            return True
        except SyntaxError:
            return False

    while i < len(lines):
        end = next((j for j in range(i + 1, min(len(lines), i + 12) + 1)
                    if parses(lines[i:j])), None)
        if end is None:
            dropped += bool(lines[i].strip())
            i += 1
            continue
        k = end  # extend through the indented block that follows (class/def bodies)
        while k < len(lines) and (lines[k][:1] in (" ", "\t") or not lines[k].strip()):
            k += 1
        end = next((j for j in range(k, end, -1) if parses(lines[i:j])), end)
        good.extend(lines[i:end])
        i = end
    return "\n".join(good), dropped


SAFE_IMPORTS = {"dataclasses", "typing", "enum", "__future__"}
BANNED_NAMES = {"eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars",
                "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit",
                "memoryview", "help", "type"}


def sim_safety(code):
    """Drop top-level statements that could do more than build the scene: imports outside
    SAFE_IMPORTS, banned builtins, dunder attribute access (other than super().__init__),
    while-loops, and calls to .run(). Classes/functions/assignments/event calls are kept.
    Returns (code, scene_var, dropped_snippets)."""
    tree = ast.parse(code)
    lines = code.splitlines()
    remove, dropped, scene_classes, scenes = set(), [], {"Scene"}, []

    def unsafe(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Import) and any(a.name.split(".")[0] not in SAFE_IMPORTS for a in n.names):
                return True
            if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] not in SAFE_IMPORTS | {"omni_sim"}:
                return True
            if isinstance(n, ast.Name) and (n.id in BANNED_NAMES or n.id.startswith("__")):
                return True  # e.g. __builtins__["open"], __import__
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and "__" in n.value:
                return True  # "{0.__init__.__globals__}".format(obj) style attribute walks
            if isinstance(n, ast.Attribute) and n.attr.startswith("__") and n.attr != "__init__":
                return True
            if isinstance(n, (ast.While, ast.AsyncFunctionDef, ast.Global, ast.Nonlocal)):
                return True
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "run":
                return True
        return False

    for node in tree.body:
        is_main = isinstance(node, ast.If) and "__name__" in ast.dump(node.test)
        omni_import = isinstance(node, ast.ImportFrom) and node.module == "omni_sim"
        if is_main or omni_import or unsafe(node):
            if not omni_import:
                dropped.append(lines[node.lineno - 1].strip()[:80])
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            remove.update(range(first, node.end_lineno + 1))
            continue
        if isinstance(node, ast.ClassDef) and any(isinstance(b, ast.Name) and b.id in scene_classes
                                                  for b in node.bases):
            scene_classes.add(node.name)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and \
                isinstance(node.value.func, ast.Name) and node.value.func.id in scene_classes and \
                len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            scenes.append(node.targets[0].id)
    kept = [ln for i, ln in enumerate(lines, 1) if i not in remove]
    return "\n".join(kept).strip(), (scenes[0] if scenes else None), dropped


def postprocess(fmt, raw, truncated=False):
    """Return (cleaned_text, validation_note)."""
    if fmt in ("text", "group_text"):
        return raw.strip(), "ok"
    if fmt in ("python", "group"):
        code, lost = salvage_python(strip_fence(raw, "python"))
        code, scene, dropped = sim_safety(code)
        if not scene:
            return code + "\n", "NO SCENE: the model did not create scene = Scene(); nothing to run"
        notes = []
        if lost:
            notes.append(f"dropped {lost} broken line(s) (bad quoting or cut off)")
        if dropped:
            notes.append(f"removed {len(dropped)} disallowed statement(s), e.g. {dropped[0]!r}")
        code = (f"from omni_sim import *\n\n{code}\n\n"
                f'if __name__ == "__main__":\n'
                f"    {scene}.run(source=SOURCE_FILE, unit=UNIT_FILE, offset=OFFSET_S, clip_s=CLIP_S)\n")
        return code, ("SALVAGED: " + "; ".join(notes)) if notes else "ok (valid simulation script)"
    if fmt == "amr":
        amr = strip_fence(raw, "amr|penman")
        note = "ok"
        try:
            import penman
            body = "\n".join(l for l in amr.splitlines() if not l.lstrip().startswith("#"))
            graphs = [g for g in re.split(r"\n\s*\n", body) if "(" in g]
            if not graphs:
                return amr + "\n", "NO GRAPH: output contains only # comment lines"
            for g in graphs:
                if g.count("(") != g.count(")"):
                    raise ValueError(f"unbalanced brackets ({g.count('(')} open, {g.count(')')} close)")
                penman.decode(g)
            note = f"ok ({len(graphs)} graph(s) parsed)"
        except ImportError:
            note = "not validated (pip install penman to enable)"
        except Exception as e:
            note = f"PENMAN PARSE ERROR: {e}"
        return amr + "\n", note
    if fmt == "svg":
        svg = strip_fence(raw, "svg|xml")
        i = svg.find("<svg")
        j = svg.rfind("</svg>")
        if i != -1 and j != -1:
            svg = svg[i:j + len("</svg>")]
        svg, repaired = repair_svg(svg)
        try:
            ET.fromstring(svg)
            if repaired or truncated:
                return svg + "\n", "REPAIRED: output was cut off at the token limit; " \
                    "closed open tags (drawing may be incomplete, try --svg-max-tokens higher)"
            return svg + "\n", "ok (valid XML)"
        except ET.ParseError as e:
            return svg + "\n", f"XML ERROR: {e}" + (" (output hit token limit)" if truncated else "")
    raise ValueError(fmt)


def add_header(fmt, body, src, unit, offset, clip_s=None):
    """Prepend the source file name (and chunk/offset) in each format's comment syntax."""
    info = f"source: {src} | unit: {unit} | offset_s: {offset:g}"
    if fmt in ("python", "group"):
        kind = "group-level simulation" if fmt == "group" else "Simulation"
        return (f"# {kind} of {unit} ({src}), written by Qwen3-Omni.\n"
                f"# Run: python <this file> [--speed 1] [--json events.json]  (needs omni_sim.py next to it)\n"
                f"SOURCE_FILE = {src!r}\nUNIT_FILE = {unit!r}\nOFFSET_S = {offset:g}\n"
                f"CLIP_S = {clip_s!r}\n\n{body}")
    if fmt == "amr":
        return f"# ::source {src} ::unit {unit} ::offset_s {offset:g}\n\n{body}"
    if fmt == "svg":
        return f"<!-- {info.replace('--', '- -')} -->\n{body}"
    return f"[{info}]\n\n{body}"


def run_simulation(path):
    """Run a whitelisted simulation script once. Returns (note, sim_txt, events_json)."""
    import sys
    stem = os.path.splitext(path)[0]
    sim_txt, ev_json = stem + ".sim.txt", stem + ".events.json"
    def limits():  # POSIX only: cap memory (2 GB) and CPU time (30 s) of the model-written script
        import resource
        for what, lim in (("RLIMIT_CPU", 30), ("RLIMIT_AS", 2 << 30)):  # RLIMIT_AS is a no-op on macOS
            try:
                resource.setrlimit(getattr(resource, what), (lim, lim))
            except Exception:
                pass
    try:
        r = subprocess.run([sys.executable, os.path.basename(path), "--json", os.path.basename(ev_json)],
                           cwd=os.path.dirname(os.path.abspath(path)), capture_output=True,
                           text=True, timeout=60, preexec_fn=limits if os.name == "posix" else None)
    except subprocess.TimeoutExpired:
        return "RUN ERROR: timed out", None, None
    with open(sim_txt, "w") as f:
        f.write(r.stdout + (("\n--- stderr ---\n" + r.stderr) if r.returncode else ""))
    if r.returncode:
        last = (r.stderr.strip().splitlines() or
                [f"script was killed (signal {-r.returncode}): CPU or memory limit"])[-1]
        return f"RUN ERROR: {last}", sim_txt, None
    n = len(json.load(open(ev_json))["events"])
    return f"ran OK, {n} events", sim_txt, ev_json


def parse_outputs(s):
    items = [x.strip().lower() for x in s.split(",") if x.strip()]
    if "all" in items:
        return list(ALL_FORMATS)
    bad = [x for x in items if x not in ALL_FORMATS]
    if bad or not items:
        raise SystemExit(f"--outputs must be from {ALL_FORMATS} or 'all' (got {s!r})")
    return [f for f in ALL_FORMATS if f in items]  # dedupe, stable order


# ---------- step 2: model ----------

def run_model(args, units, formats, use_audio, handle, clip_lens):
    """Load the model once; for each (video, offset) unit and format, generate and call
    handle(unit_index, fmt, raw_text)."""
    from vllm import LLM, SamplingParams
    from transformers import Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        limit_mm_per_prompt={"video": 1, "audio": 1},
    )
    # text wants some freedom; code-like formats are better near-deterministic
    sp = {
        f: SamplingParams(temperature=0.2 if f in ("text", "group_text") else 0.1,
                          max_tokens=args.svg_max_tokens if f == "svg" else args.max_tokens,
                          repetition_penalty=args.repetition_penalty)
        for f in formats
    }

    for ui, (video, _offset) in enumerate(units):
        wav = extract_wav(video, os.path.dirname(video) or ".") if use_audio else None
        if use_audio and wav is None:
            print(f"{video}: no audio stream; video-only.")
        for fmt in formats:
            content = [{"type": "video", "video": video}]
            if wav:
                content.append({"type": "audio", "audio": wav})
            clip = clip_lens[ui]
            prompt = PROMPTS[fmt].replace("{CLIP}", f"{clip:.0f}" if clip else "?")
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
            mm_data = {"video": videos}
            if audios:
                mm_data["audio"] = audios
            inputs = {
                "prompt": text,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": {"use_audio_in_video": False},
            }
            out = llm.generate([inputs], sampling_params=sp[fmt])[0]
            raw = out.outputs[0].text
            truncated = out.outputs[0].finish_reason == "length"
            if truncated:
                print(f"WARNING: {video} [{fmt}] hit the max_tokens limit; output is cut off.")
            print(f"\n=== {video} [{fmt}] ===\n{raw}\n")
            handle(ui, fmt, raw, truncated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True,
                    help="a video file, or a directory of .mp4 chunks (e.g. chunks/)")
    ap.add_argument("--start", help="clip start (seconds or [HH:]MM:SS); omit to keep original")
    ap.add_argument("--end", help="clip end (seconds or [HH:]MM:SS); omit to keep original")
    ap.add_argument("--chunk", action="store_true",
                    help="split the (clipped) video into overlapping chunks first; "
                         "off by default so the video is left untouched")
    ap.add_argument("--chunk-len", type=int, default=80)
    ap.add_argument("--stride", type=int, default=75)
    ap.add_argument("--accurate", action="store_true",
                    help="re-encode (libx264) for frame-exact cuts; default is stream copy")
    ap.add_argument("--outputs", default="text",
                    help="comma list of text,group_text,group,python,amr,svg or 'all' (default: text)")
    ap.add_argument("--out-dir", default="omni_out")
    ap.add_argument("--no-audio", action="store_true", help="video only, skip wav extraction")
    ap.add_argument("--tp", type=int, default=4, help="tensor_parallel_size")
    ap.add_argument("--max-model-len", type=int, default=49152)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--svg-max-tokens", type=int, default=8192,
                    help="separate (larger) output limit for svg, which is verbose")
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="try 1.05 if outputs loop (default 1.0 = off)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-num-batched-tokens", type=int, default=49152)
    args = ap.parse_args()

    formats = parse_outputs(args.outputs)
    os.makedirs(args.out_dir, exist_ok=True)
    start, end = parse_time(args.start), parse_time(args.end)

    if os.path.isdir(args.video):
        if start is not None or end is not None or args.chunk:
            raise SystemExit("--start/--end/--chunk need a single video file, not a directory")
        units = units_from_dir(args.video)
    elif os.path.isfile(args.video):
        video, offset = maybe_clip(args.video, start, end, args.out_dir, args.accurate)
        if args.chunk:
            units = make_chunks(video, offset, args.out_dir, args.chunk_len, args.stride,
                                args.accurate)
        else:
            units = [(video, offset)]
            if duration(video) > args.chunk_len + 10:
                print(f"WARNING: video is {duration(video):.0f}s; long inputs may exceed "
                      f"--max-model-len. Consider --chunk or --start/--end.")
    else:
        raise SystemExit(f"Not found: {args.video}")

    summary = {"input": args.video, "input_name": os.path.basename(os.path.normpath(args.video)), "units": []}
    for video, offset in units:
        summary["units"].append({"model_input": video, "offset_s": offset, "outputs": {}})

    def safe_duration(v):
        try:
            return duration(v)
        except Exception:
            return None
    clip_lens = [safe_duration(v) for v, _ in units]
    if "python" in formats or "group" in formats:
        shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "omni_sim.py"),
                    args.out_dir)

    src_name = os.path.basename(os.path.normpath(args.video))
    src_stem = os.path.splitext(src_name)[0]

    def handle(ui, fmt, raw, truncated):
        video, offset = units[ui]
        cleaned, note = postprocess(fmt, raw, truncated)
        if truncated and fmt != "svg":
            note = "CUT OFF at token limit; " + note
        base = os.path.splitext(os.path.basename(video))[0]
        name = base if base.startswith(src_stem) else f"{src_stem}_{base}"
        path = os.path.join(args.out_dir, f"{name}.{EXT[fmt]}")
        with open(path, "w") as f:
            f.write(add_header(fmt, cleaned, src_name, os.path.basename(video), offset,
                               clip_lens[ui]))
        entry = {"file": path, "check": note, "truncated": truncated}
        if fmt in ("python", "group") and "NO SCENE" not in note:
            run_note, sim_txt, ev_json = run_simulation(path)
            note = f"{note}; {run_note}"
            entry.update(check=note, sim_output=sim_txt, events_json=ev_json)
        summary["units"][ui]["source_name"] = src_name
        summary["units"][ui]["outputs"][fmt] = entry
        print(f"[{fmt}] {path}  -> {note}")
        # rewrite after every result so a crash keeps what was done
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    run_model(args, units, formats, not args.no_audio, handle, clip_lens)


if __name__ == "__main__":
    main()
