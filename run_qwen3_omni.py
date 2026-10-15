"""
Batch-process video chunks through Qwen3-Omni-30B-A3B-Instruct via vLLM's offline API.
Uses the exact message/processor pattern from the official Qwen3-Omni README.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_qwen3_omni.py --chunk-dir chunks --out chunk_outputs.jsonl
"""
import argparse
import glob
import json
import os

from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# Edit this to match your paper's schema (participant/behavior/modality/start/end/target/confidence)
# once you've confirmed the basic pipeline runs end to end.
PROMPT = (
    "Analyze this video clip in detail, covering both audio and visual behavior:\n"
    "1. TRANSCRIPT: Transcribe what is said, word for word, labeling who speaks "
    "each line (e.g. 'woman on the left'). If no speech occurs in a span, say so.\n"
    "2. NON-VERBAL BEHAVIOR: Describe gestures, gaze direction, posture, and "
    "other notable non-verbal behavior for each participant.\n"
    "Give timestamps relative to the start of this clip (mm:ss) throughout."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", default="chunks")
    ap.add_argument("--out", default="chunk_outputs.jsonl")
    ap.add_argument("--tp", type=int, default=4, help="tensor_parallel_size")
    ap.add_argument("--max-model-len", type=int, default=49152)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-num-batched-tokens", type=int, default=49152)
    args = ap.parse_args()

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
    sampling_params = SamplingParams(temperature=0.2, max_tokens=args.max_tokens)

    chunk_files = sorted(glob.glob(os.path.join(args.chunk_dir, "*.mp4")))
    if not chunk_files:
        raise SystemExit(f"No .mp4 files found in {args.chunk_dir}")

    print(f"Found {len(chunk_files)} chunks. Model loaded, starting inference...")

    results = []
    for cf in chunk_files:
        wav_path = os.path.splitext(cf)[0] + ".wav"
        content = [{"type": "video", "video": cf}]
        if os.path.exists(wav_path):
            content.append({"type": "audio", "audio": wav_path})
        content.append({"type": "text", "text": PROMPT})

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

        out = llm.generate([inputs], sampling_params=sampling_params)[0]
        output_text = out.outputs[0].text

        print(f"\n=== {cf} ===\n{output_text}\n")
        results.append({"chunk": cf, "text": output_text})

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
