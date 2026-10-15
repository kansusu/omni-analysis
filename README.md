# Omni group & individual behavior annotation

Run [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) on a video of a small group
and get its behavior back in several forms: plain text, a runnable Python **simulation** of the clip,
AMR graphs and SVG, at both the **individual** level (what each person does) and the **group** level
(what two or more people do together).

```
video ──► optional clip (--start/--end) ──► 20–80 s chunks ──► Qwen3-Omni (vLLM)
                                                               │  one prompt per output format
        ┌──────────────────────────────────────────────────────┘
        ▼
  text            <chunk>.txt          transcript + per-person behavior
  group_text      <chunk>.group.txt    group-level events as text
  python          <chunk>.py           simulation of individual events   ─┐ run once →
  group           <chunk>.group.py     simulation of group-level events   ─┘ .sim.txt / .events.json
  amr, svg        <chunk>.amr / .svg   (experimental, see "Known limits")
        ▼
  visualize_omni_outputs.py  ──►  report.html   (health grid, timelines, tables)
```

## Files

| File | Purpose |
|---|---|
| `run_qwen3_omni_multi.py` | Main runner. Clip, chunk, prompt the model, clean/validate/save every output. |
| `omni_sim.py` | Runtime for the model-written simulation scripts: `Scene`, `Person`, `Thing`, `Group`, `Utterance`, `Gaze`, `JointAttention`, … plus group-event detection. |
| `visualize_omni_outputs.py` | Builds one self-contained HTML report from an output folder. Standard library only. |
| `evaluate_sim_outputs.py` | Quality report over the simulation outputs (runs? classes? repeats? timing?). |
| `chunk_video.sh`, `extract_audio.sh` | The original ffmpeg helpers (the runner does the same thing itself). |
| `run_qwen3_omni.py`, `run_qwen3_omni_video_only_WORKING.py`, `test_audio_single.py` | The original single-purpose scripts. |
| `tests/` | Unit tests, no GPU needed. |

## Quick start

```bash
# on the GPU machine
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_qwen3_omni_multi.py \
    --video talk.mp4 --chunk --chunk-len 20 --stride 20 \
    --outputs text,group_text,python,group --out-dir omni_out

python visualize_omni_outputs.py omni_out --chunk-len 20      # → omni_out/report.html
python omni_out/<chunk>.py --speed 1                          # replay one simulation in real time
```

Useful options: `--start/--end` (clip only a span; default is to leave the video untouched),
`--chunk` (split into chunks), `--outputs` (any combination of
`text, group_text, python, group, amr, svg, all`), `--repetition-penalty 1.05` (if outputs loop),
`--max-tokens`, `--svg-max-tokens`.

## The simulation format

The model writes ordinary Python on top of `omni_sim`, so people and events are objects that later tools can extend
(for example, an `Utterance.render()` that calls a TTS module):

```python
from omni_sim import *
scene = Scene()
p1 = Person(scene, "p1", appearance="short hair", clothing_color="blue", voice_type="low male")
p2 = Person(scene, "p2", clothing_color="green")
blocks = Thing(scene, "blocks", description="pile of colored blocks")

p1.say(0.0, 2.5, "Put the red one on the left.", to=p2)
p2.look(0.5, 3.0, at=p1)
scene.group(p1, p2).look(5, 9, at=blocks)          # group event: joint attention
```

Group-level events (`JointAttention`, `MutualGaze`, `JointGesture`, `MutualGesture`, `JointAction`) are either
stated by the model (`--outputs group`) or **derived** from individual events: overlapping gaze/gesture events
from 2+ people with the same target (joint) or aimed at each other (mutual).

### Safety of model-written code

The runner never executes model output directly. Statements are salvaged if broken, then filtered
(no imports other than `dataclasses/typing/enum`, no `open/eval/exec/getattr/type`, no dunder access, no `while`,
no `scene.run()`), and the script runs once in a subprocess with a 60 s wall-clock, 30 s CPU and (Linux) 2 GB
memory limit. This is a blocklist, **not a sandbox**: run it on machines where that is acceptable.

## Known limits (from our runs on a 5-minute, 4-person video)

* Output loops are the main failure: the model repeats one line until the token limit. Use short chunks,
  `--repetition-penalty`, and treat repeated lines as one event.
* Group labels need checking: "mutual gaze" is often used for several people looking at a third person
  (that is joint attention), and group targets sometimes include a group member.
* AMR and SVG straight from video are mostly unusable; derive them from the text or the events instead.
* Participant ids (`p1`, `p2`) are assigned separately in every chunk.

## Data

Videos, transcripts, generated reports and model outputs are excluded by `.gitignore` because they contain
recordings and speech of study participants. Do not commit them.

## Tests

```bash
python -m unittest discover -s tests -v
```
