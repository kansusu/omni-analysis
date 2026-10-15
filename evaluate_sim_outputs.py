"""
Score the python SIMULATION outputs of run_qwen3_omni_multi.py: how many generations
worked, what failed and why, and how good the working ones look.

Reads the output folder only (summary.json, <name>.py, <name>.events.json, <name>.sim.txt).
It never runs the generated code; the .py files are inspected with ast.

Usage:
    python evaluate_sim_outputs.py omni_out_sim
    python evaluate_sim_outputs.py omni_out_sim --csv sim_eval.csv
"""
import argparse
import ast
import collections
import csv
import glob
import json
import os
import re

BASES = {"Person", "Thing", "Entity", "Event", "Utterance", "Gesture", "Gaze", "Posture",
         "Action", "Scene"}
ACTIONS = {"say", "gesture", "look", "posture", "act", "do"}


def status_of(check):
    if not check:
        return "NOT RUN"
    if "NO SCENE" in check:
        return "NO SCENE"
    if "RUN ERROR" in check:
        return "RUN ERROR"
    if check.startswith("SALVAGED"):
        return "SALVAGED"
    return "OK"


def inspect_code(path):
    """Static look at the generated script: classes, attributes, event calls."""
    out = {"classes": [], "person_attrs": collections.Counter(), "event_calls": 0}
    try:
        tree = ast.parse(open(path).read())
    except (OSError, SyntaxError):
        return out
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            bases = [b.id for b in n.bases if isinstance(b, ast.Name)]
            out["classes"].append(f"{n.name}({', '.join(bases)})")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ACTIONS:
            out["event_calls"] += 1
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.args:
            out["person_attrs"].update(k.arg for k in n.keywords if k.arg and k.arg != "description")
    return out


def quality(ev_path):
    """Metrics from the events the simulation actually produced."""
    d = json.load(open(ev_path))
    evs, clip = d.get("events", []), d.get("clip_s")
    people = {p["id"] for p in d.get("participants", [])}
    n = len(evs)
    q = {"people": len(people), "things": len(d.get("things", [])), "events": n,
         "modalities": dict(collections.Counter(e.get("modality") for e in evs)),
         "event_types": dict(collections.Counter(e.get("type", "?") for e in evs))}
    if not n:
        return q
    contents = [re.sub(r"\W+", " ", str(e.get("content", "")).lower()).strip() for e in evs]
    q.update(
        distinct_starts=len({e["start"] for e in evs}),
        pct_outside_clip=round(100 * sum(e["end"] > clip + 0.5 for e in evs) / n) if clip else None,
        pct_repeated_content=round(100 * (1 - len(set(contents)) / n)),
        pct_with_target=round(100 * sum(bool(e.get("target")) for e in evs) / n),
        pct_person_to_person=round(100 * sum(e.get("target") in people for e in evs) / n),
        coverage_s=round(max(e["end"] for e in evs) - min(e["start"] for e in evs), 1),
        clip_s=clip,
    )
    return q


def problems(row):
    """Plain-language warnings for a chunk that ran."""
    p = []
    if row.get("events", 0) == 0:
        p.append("no events recorded")
        return p
    if row.get("distinct_starts", 0) <= 2 and row["events"] >= 5:
        p.append("timestamps degenerate (almost all events start at the same time)")
    if (row.get("pct_outside_clip") or 0) > 10:
        p.append(f"{row['pct_outside_clip']}% of events run past the clip")
    if row.get("pct_repeated_content", 0) > 40:
        p.append(f"{row['pct_repeated_content']}% of event texts are repeats (looping)")
    if row.get("clip_s") and row.get("coverage_s", 0) < 0.5 * row["clip_s"]:
        p.append(f"events only cover {row['coverage_s']}s of a {row['clip_s']:g}s clip")
    if not row.get("custom_classes"):
        p.append("no custom classes: the model did not build a class hierarchy")
    if not row.get("person_attrs"):
        p.append("people have no attributes (appearance, clothing_color, voice_type ...)")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--csv", help="also write one row per chunk to this CSV")
    args = ap.parse_args()

    checks = {}
    sp = os.path.join(args.dir, "summary.json")
    if os.path.exists(sp):
        for u in json.load(open(sp)).get("units", []):
            o = u.get("outputs", {}).get("python")
            if o:
                checks[os.path.splitext(os.path.basename(o["file"]))[0]] = (o.get("check", ""), o.get("truncated"))

    rows = []
    for py in sorted(glob.glob(os.path.join(args.dir, "*.py"))):
        stem = os.path.splitext(os.path.basename(py))[0]
        if stem == "omni_sim" or stem.endswith(".group"):  # group simulations are scored separately
            continue
        check, truncated = checks.get(stem, ("", None))
        code = inspect_code(py)
        row = {"chunk": stem, "status": status_of(check), "truncated": truncated, "check": check,
               "custom_classes": [c for c in code["classes"]],
               "person_attrs": sorted(code["person_attrs"]), "event_calls_in_code": code["event_calls"]}
        m = re.search(r"dropped (\d+) broken", check)
        row["dropped_lines"] = int(m.group(1)) if m else 0
        m = re.search(r"removed (\d+) disallowed", check)
        row["removed_statements"] = int(m.group(1)) if m else 0
        ev = os.path.join(args.dir, stem + ".events.json")
        if os.path.exists(ev):
            row.update(quality(ev))
        row["problems"] = problems(row) if row["status"] in ("OK", "SALVAGED") else []
        rows.append(row)

    if not rows:
        raise SystemExit(f"No simulation .py files in {args.dir}")

    # ---------------- report ----------------
    st = collections.Counter(r["status"] for r in rows)
    print(f"\n=== Generation success ({len(rows)} chunks) ===")
    for k in ("OK", "SALVAGED", "RUN ERROR", "NO SCENE", "NOT RUN"):
        if st[k]:
            print(f"  {k:<10} {st[k]:>3}  ({100 * st[k] // len(rows)}%)")
    ran = [r for r in rows if r["status"] in ("OK", "SALVAGED")]
    print(f"  -> runnable simulations: {len(ran)}/{len(rows)}")
    print(f"  cut off at token limit: {sum(1 for r in rows if r['truncated'])}")
    print(f"  broken lines dropped: {sum(r['dropped_lines'] for r in rows)}   "
          f"unsafe statements removed: {sum(r['removed_statements'] for r in rows)}")

    print("\n=== Per chunk ===")
    print(f"  {'chunk':<38} {'status':<10} {'events':>6} {'people':>6} {'classes':>7} {'attrs':>5}  problems")
    for r in rows:
        print(f"  {r['chunk'][-38:]:<38} {r['status']:<10} {r.get('events', '-'):>6} "
              f"{r.get('people', '-'):>6} {len(r['custom_classes']):>7} {len(r['person_attrs']):>5}  "
              f"{len(r['problems']) or ''}")

    print("\n=== Failures and problems ===")
    any_p = False
    for r in rows:
        if r["status"] not in ("OK",) or r["problems"]:
            any_p = True
            print(f"  {r['chunk']}: {r['status']}")
            if r["status"] != "OK":
                print(f"      check: {r['check'][:200]}")
            for p in r["problems"]:
                print(f"      - {p}")
    if not any_p:
        print("  none")

    if ran:
        print("\n=== What the model built (runnable chunks) ===")
        classes = collections.Counter(c for r in ran for c in r["custom_classes"])
        attrs = collections.Counter(a for r in ran for a in r["person_attrs"])
        mods, types = collections.Counter(), collections.Counter()
        for r in ran:
            mods.update(r.get("modalities", {}))
            types.update(r.get("event_types", {}))
        print(f"  custom classes: {dict(classes) or 'none'}")
        print(f"  entity attributes used: {dict(attrs) or 'none'}")
        print(f"  events by modality: {dict(mods)}")
        print(f"  events by class: {dict(types)}")
        tot = sum(r.get("events", 0) for r in ran)
        if tot:
            w = lambda k: round(sum(r.get(k, 0) * r.get("events", 0) for r in ran) / tot)
            print(f"  events: {tot} total; {w('pct_with_target')}% have a target, "
                  f"{w('pct_person_to_person')}% are aimed at another person, "
                  f"{w('pct_repeated_content')}% repeated text")

    if args.csv:
        keys = ["chunk", "status", "truncated", "events", "people", "things", "distinct_starts",
                "coverage_s", "clip_s", "pct_outside_clip", "pct_repeated_content", "pct_with_target",
                "pct_person_to_person", "dropped_lines", "removed_statements", "custom_classes",
                "person_attrs", "problems", "check"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: ("; ".join(map(str, r[k])) if isinstance(r.get(k), list) else r.get(k))
                            for k in keys})
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
