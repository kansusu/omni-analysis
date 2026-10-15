"""
Tiny runtime for the "video simulation" scripts that Qwen3-Omni writes.

The model describes a clip as a scene made of objects: people and things (Entity
subclasses) that produce timed events (Event subclasses) aimed at each other. Running the
generated script replays the events on a simulated clock, prints them in time order, and
summarises who interacted with whom.

Class hierarchy (the model may subclass any of these and add attributes/methods):

    Scene                     holds entities and events, replays them (run)
    Entity(scene, id, description="", **attributes)
     |- Person                say / gesture / look / posture / act -> Event objects
     |- Thing                 objects people use (blocks, tablet, ...)
     '- Group(scene, id, members)   2+ people acting together -> GroupEvent objects
    Event(actor, start, end, content="", target=None)
     |- Utterance             speech; .text, .voice; render() is the hook for a TTS module
     |- Gesture
     |- Gaze
     |- Posture
     |- Action                any other physical action
     '- GroupEvent            an event performed by a Group (all members are actors)
         |- JointAttention    2+ people look at the same person/thing at the same time
         |- MutualGaze        people look at each other
         |- JointGesture      2+ people point/gesture at the same person/thing
         |- MutualGesture     people point/gesture at each other
         '- JointAction       2+ people do something together (e.g. on the same object)

Group-level events come from two places:
  * stated: the script records them, e.g.
        g = scene.group(p1, p3)
        g.look(2, 6, at=blocks)                 # joint attention
        g.look_at_each_other(6, 8)              # mutual gaze
        g.gesture(3, 5, "point", at=p2)         # both point at the same person
        g.gesture_at_each_other(9, 10, "point")
        g.act(10, 14, "lift the tower together", on=blocks)
  * derived: Scene.derive_group_events() finds them in individual events: overlapping
    gaze/gesture events of 2+ people with the same target, or aimed at each other.
    The replay prints both, and the JSON export has them under "derived_group_events".

    from omni_sim import *
    scene = Scene()
    class Builder(Person):
        role = "builder"
    p1 = Person(scene, "p1", appearance="short hair", clothing_color="blue", voice_type="low male")
    p2 = Builder(scene, "p2", clothing_color="green")
    blocks = Thing(scene, "blocks", description="pile of colored blocks")
    p1.say(0.0, 2.5, "Put the red one on the left.", to=p2)
    p2.look(0.5, 3.0, at=p1)
    p2.gesture(2.0, 4.0, "points at the left corner", at=blocks)
    p2.act(4.0, 7.0, "places a red block", on=blocks)

Run a generated script:
    python clip.py                  # print instantly
    python clip.py --speed 1        # replay in real time (2 = twice as fast)
    python clip.py --json out.json  # also export events for visualize_omni_outputs.py

Standard library only.
"""
import argparse
import collections
import json
import sys
import time

__all__ = ["Scene", "Entity", "Person", "Thing", "Group",
           "Event", "Utterance", "Gesture", "Gaze", "Posture", "Action",
           "GroupEvent", "JointAttention", "MutualGaze", "JointGesture", "MutualGesture",
           "JointAction"]


def _t(x):
    try:
        return max(0.0, float(x))
    except (TypeError, ValueError):
        return 0.0


def _mmss(s):
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


def _plain(v):
    """JSON-safe version of an attribute value."""
    if isinstance(v, Entity):
        return v.id
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return str(v)


# =============================== events ===============================

class Event:
    """Something an actor does over [start, end] seconds of the clip, optionally aimed at a target."""
    modality = "other"
    verb, prep = "acts", "on"

    def __init__(self, actor, start, end, content="", target=None, **attributes):
        a, b = _t(start), _t(end)
        self.start, self.end = (a, b) if b >= a else (b, a)
        self.actor, self.content, self.target = actor, str(content or ""), target
        self.attributes = attributes

    @property
    def duration(self):
        return self.end - self.start

    def render(self):
        """Human-readable form used by the replay. Subclasses / later tools can override."""
        return self.content

    def to_dict(self):
        t = self.target
        return {
            "type": type(self).__name__, "start": self.start, "end": self.end,
            "participant": self.actor.id if isinstance(self.actor, Entity) else str(self.actor),
            "modality": self.modality, "content": self.content,
            "target": t.id if isinstance(t, Entity) else (str(t) if t else None),
            **({"attributes": _plain(self.attributes)} if self.attributes else {}),
        }


class Utterance(Event):
    """Speech. `render()` is the place to plug in a TTS module later."""
    modality, verb, prep = "speech", "says", "to"

    @property
    def text(self):
        return self.content

    @property
    def voice(self):
        return getattr(self.actor, "voice_type", None)

    def render(self):
        return f'"{self.text}"'


class Gesture(Event):
    modality, verb, prep = "gesture", "gestures", "at"


class Gaze(Event):
    modality, verb, prep = "gaze", "looks", "at"

    def render(self):
        return "" if self.content in ("", f"looks at {self.target}") else self.content


class GroupEvent(Event):
    """An event performed together by every member of a Group."""

    @property
    def members(self):
        return list(getattr(self.actor, "members", []))

    def render(self):
        return "" if self.content.startswith(("derived", "look ")) else self.content

    def to_dict(self):
        d = super().to_dict()
        d["participants"] = [m.id for m in self.members]
        d["group"] = True
        return d


class JointAttention(GroupEvent):
    modality, verb, prep = "gaze", "jointly look", "at"


class MutualGaze(GroupEvent):
    modality, verb, prep = "gaze", "look at each other", ""


class JointGesture(GroupEvent):
    modality, verb, prep = "gesture", "jointly gesture", "at"


class MutualGesture(GroupEvent):
    modality, verb, prep = "gesture", "gesture at each other", ""


class JointAction(GroupEvent):
    modality, verb, prep = "other", "act together", "on"


class Posture(Event):
    modality, verb, prep = "posture", "posture", ""


class Action(Event):
    modality, verb, prep = "other", "acts", "on"


# =============================== entities ===============================

class Entity:
    """Anything in the scene. Extra keyword arguments become attributes
    (e.g. appearance=..., clothing_color=..., voice_type=...)."""
    kind = "entity"

    def __init__(self, scene, id, description="", **attributes):
        self.scene, self.id, self.description = scene, str(id), str(description or "")
        self.attributes = dict(attributes)
        for k, v in attributes.items():
            if not hasattr(type(self), k):
                setattr(self, k, v)
        if scene is not None:
            scene.add(self)

    def summary(self):
        """Short description: the given description, else the attributes."""
        if self.description:
            return self.description
        return ", ".join(f"{k}={v}" for k, v in self.attributes.items() if v not in (None, ""))

    def to_dict(self):
        return {"id": self.id, "class": type(self).__name__, "description": self.summary(),
                "attributes": _plain(self.attributes)}

    def __repr__(self):
        return self.id


class Thing(Entity):
    kind = "thing"


class Person(Entity):
    kind = "person"

    def __init__(self, scene, id, description="", **attributes):
        self.events = []
        super().__init__(scene, id, description, **attributes)

    def do(self, event):
        """Record any Event (including custom Event subclasses) for this person."""
        self.events.append(event)
        if self.scene is not None:
            self.scene.events.append(event)
        return event

    def say(self, start, end, text, to=None, **kw):
        return self.do(Utterance(self, start, end, text, to, **kw))

    def gesture(self, start, end, what, at=None, **kw):
        return self.do(Gesture(self, start, end, what, at, **kw))

    def look(self, start, end, at=None, what="", **kw):
        return self.do(Gaze(self, start, end, what or f"looks at {at}", at, **kw))

    def posture(self, start, end, what, **kw):
        return self.do(Posture(self, start, end, what, None, **kw))

    def act(self, start, end, what, on=None, **kw):
        return self.do(Action(self, start, end, what, on, **kw))

    # aliases the model tends to use
    def gaze(self, start, end, at=None, what="", **kw):
        return self.look(start, end, at=at, what=what, **kw)

    def point(self, start, end, at=None, what="points", **kw):
        return self.gesture(start, end, what, at=at, **kw)


class Group(Entity):
    """Two or more people acting together; its events have every member as an actor."""
    kind = "group"

    def __init__(self, scene, id, members, description="", **attributes):
        self.members = [m for m in members if isinstance(m, Entity)]
        super().__init__(scene, id or "+".join(m.id for m in self.members), description, **attributes)

    def summary(self):
        return self.description or " + ".join(m.id for m in self.members)

    def to_dict(self):
        d = super().to_dict()
        d["members"] = [m.id for m in self.members]
        return d

    def do(self, event):
        if self.scene is not None:
            self.scene.events.append(event)
        return event

    def look(self, start, end, at=None, what="", **kw):
        """Joint attention: every member looks at the same target."""
        return self.do(JointAttention(self, start, end, what or f"look at {at}", at, **kw))

    def look_at_each_other(self, start, end, what="", **kw):
        return self.do(MutualGaze(self, start, end, what or "look at each other", None, **kw))

    def gesture(self, start, end, what, at=None, **kw):
        """Every member points/gestures at the same target."""
        return self.do(JointGesture(self, start, end, what, at, **kw))

    def gesture_at_each_other(self, start, end, what="point", **kw):
        return self.do(MutualGesture(self, start, end, what, None, **kw))

    def act(self, start, end, what, on=None, **kw):
        return self.do(JointAction(self, start, end, what, on, **kw))

    gaze = look
    gaze_at_each_other = look_at_each_other

    def point(self, start, end, at=None, what="point", **kw):
        return self.gesture(start, end, what, at=at, **kw)

    def point_at_each_other(self, start, end, what="point", **kw):
        return self.gesture_at_each_other(start, end, what, **kw)


# =============================== scene ===============================

class Scene:
    def __init__(self, duration=None, **attributes):
        self.duration = _t(duration) if duration is not None else None
        self.attributes = attributes
        self.people, self.things, self.groups, self.events = {}, {}, {}, []

    def add(self, entity):
        if isinstance(entity, Group):
            self.groups[entity.id] = entity
        else:
            (self.people if isinstance(entity, Person) else self.things)[entity.id] = entity
        return entity

    def group(self, *members, id=None, description="", cls=None, **attributes):
        """scene.group(p1, p3) -> a Group whose events are done by both."""
        if len(members) == 1 and isinstance(members[0], (list, tuple)):
            members = members[0]
        return (cls or Group)(self, id, list(members), description, **attributes)

    # --------------------------------------------------------- group detection
    def _target_of(self, e):
        """Event target, or an entity named in its text ("points at blocks")."""
        if e.target is not None:
            return e.target
        words = set(e.content.lower().replace(",", " ").replace(".", " ").split())
        for ent in list(self.things.values()) + list(self.people.values()):
            if ent is not e.actor and ent.id.lower() in words:
                return ent
        return None

    def derive_group_events(self, min_overlap=0.5):
        """Group-level events implied by individual gaze/gesture events:
        2+ people aimed at the same target at overlapping times (joint), or
        two people aimed at each other at overlapping times (mutual)."""
        indiv = [e for e in self.events
                 if isinstance(e.actor, Person) and not isinstance(e, GroupEvent)]
        out = []
        kinds = (("gaze", JointAttention, MutualGaze), ("gesture", JointGesture, MutualGesture))
        for modality, joint_cls, mutual_cls in kinds:
            evs = [(e, self._target_of(e)) for e in indiv if e.modality == modality]
            evs = [(e, t) for e, t in evs if t is not None]
            # joint: same target, 2+ different actors active over the same span
            by_target = collections.defaultdict(list)
            for e, t in evs:
                by_target[t.id if isinstance(t, Entity) else str(t)].append((e, t))
            for _, es in by_target.items():
                bounds = sorted({x for e, _ in es for x in (e.start, e.end)})
                runs = []
                for a, b in zip(bounds, bounds[1:]):
                    actors = frozenset(e.actor for e, _ in es if e.start <= a and e.end >= b)
                    if len(actors) < 2:
                        continue
                    if runs and runs[-1][2] == actors and runs[-1][1] == a:
                        runs[-1][1] = b
                    else:
                        runs.append([a, b, actors])
                for a, b, actors in runs:
                    if b - a >= min_overlap:
                        g = Group(None, None, sorted(actors, key=lambda p: p.id))
                        out.append(joint_cls(g, a, b, f"derived from {modality} events",
                                             es[0][1], derived=True))
            # mutual: A aimed at B while B aimed at A
            seen = set()
            for i, (x, tx) in enumerate(evs):
                for y, ty in evs[i + 1:]:
                    if tx is y.actor and ty is x.actor and x.actor is not y.actor:
                        a, b = max(x.start, y.start), min(x.end, y.end)
                        key = (frozenset((x.actor.id, y.actor.id)), a, b)
                        if b - a >= min_overlap and key not in seen:
                            seen.add(key)
                            g = Group(None, None, sorted([x.actor, y.actor], key=lambda p: p.id))
                            out.append(mutual_cls(g, a, b, f"derived from {modality} events",
                                                  None, derived=True))
        return sorted(out, key=lambda e: (e.start, str(e.actor)))

    # factories, kept for convenience
    def person(self, id, description="", cls=None, **attributes):
        return (cls or Person)(self, id, description, **attributes)

    def thing(self, id, description="", cls=None, **attributes):
        return (cls or Thing)(self, id, description, **attributes)

    # ------------------------------------------------------------------ export
    def to_dict(self, source=None, unit=None, offset=0.0, clip_s=None):
        return {
            "source": source, "unit": unit, "offset_s": offset,
            "clip_s": clip_s or self.duration,
            "participants": [p.to_dict() for p in self.people.values()],
            "things": [t.to_dict() for t in self.things.values()],
            "groups": [g.to_dict() for g in self.groups.values()],
            "events": [e.to_dict() for e in self.sorted_events()],
            "derived_group_events": [e.to_dict() for e in self.derive_group_events()],
        }

    def sorted_events(self):
        return sorted(self.events, key=lambda e: (e.start, getattr(e.actor, "id", "")))

    # --------------------------------------------------------------------- run
    def run(self, source=None, unit=None, offset=0.0, clip_s=None, argv=None):
        ap = argparse.ArgumentParser(description="Replay a simulated video clip.")
        ap.add_argument("--speed", type=float, default=0,
                        help="0 = print instantly (default), 1 = real time, 2 = 2x ...")
        ap.add_argument("--json", help="write events as JSON to this path")
        args = ap.parse_args(sys.argv[1:] if argv is None else argv)

        clip = clip_s or self.duration
        evs = self.sorted_events()
        out = sys.stdout

        out.write(f"=== Simulating {unit or 'clip'}"
                  f"{' from ' + source if source else ''} "
                  f"(video time {_mmss(offset)}"
                  f"{' - ' + _mmss(offset + clip) if clip else ''}) ===\n")
        for e in list(self.people.values()) + list(self.things.values()) + list(self.groups.values()):
            out.write(f"  {e.kind:<6} {e.id} [{type(e).__name__}]: {e.summary()}\n")
        out.write(f"  {len(evs)} events\n\n")

        def line(e):
            tgt = f" {e.prep or '->'} {e.target}" if e.target else ""
            body = e.render()
            flag = "  [outside clip]" if clip and e.end > clip + 0.5 else ""
            return (f"[t={e.start:6.1f}s | {_mmss(offset + e.start)}] "
                    f"{e.actor} {e.verb}{tgt}{': ' + body if body else ''}  "
                    f"({e.duration:.1f}s){flag}\n")

        clock = 0.0
        for e in evs:
            if args.speed > 0 and e.start > clock:
                time.sleep((e.start - clock) / args.speed)
            clock = max(clock, e.start)
            out.write(line(e))
            out.flush()

        if any(isinstance(e.actor, Person) for e in evs):  # only meaningful with individual events
            derived = self.derive_group_events()
            out.write("\n--- group-level events derived from individual events ---\n")
            for e in derived:
                out.write(line(e))
            if not derived:
                out.write("  none (no two people aimed gaze/gestures at the same target or at each other)\n")

        # ---- summary: activity per person and who-addresses-whom
        out.write("\n--- summary ---\n")
        per = collections.defaultdict(collections.Counter)
        for e in evs:
            per[str(e.actor)][type(e).__name__] += 1
        for pid in list(self.people) + [k for k in per if k not in self.people]:
            c = per.get(pid, {})
            if not c:
                continue
            out.write(f"  {pid:<4} " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) + "\n")
        links = collections.Counter((str(e.actor), str(e.target), type(e).__name__)
                                    for e in evs if isinstance(e.target, Person))
        if links:
            out.write("  interactions between people:\n")
            for (a, b, m), n in sorted(links.items()):
                out.write(f"    {a} -> {b}: {n} x {m}\n")
        if clip:
            late = sum(1 for e in evs if e.end > clip + 0.5)
            if late:
                out.write(f"  WARNING: {late} event(s) run past the {clip:g}s clip\n")

        data = self.to_dict(source, unit, offset, clip)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(data, f, indent=2)
        return data
