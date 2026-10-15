"""
Single-chunk smoke test to confirm whether audio is actually reaching the model.
Uses a prompt that can only be answered correctly if audio data is present.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python test_audio_single.py chunks/chunk_000_start00000.mp4
"""
import sys

from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

PROMPT = (
    "Transcribe exactly what is said in this clip, word for word, in order. "
    "Note who appears to be speaking (e.g. 'woman on the left') for each line. "
    "If you cannot hear any speech or audio at all, say so explicitly instead of guessing."
)


def main():
    chunk_path = sys.argv[1] if len(sys.argv) > 1 else "chunks/chunk_000_start00000.mp4"
    wav_path = chunk_path.rsplit(".", 1)[0] + ".wav"

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        tensor_parallel_size=4,
        dtype="bfloat16",
        max_model_len=49152,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=49152,
        limit_mm_per_prompt={"video": 1, "audio": 1},
    )
    sampling_params = SamplingParams(temperature=0.2, max_tokens=1024)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": chunk_path},
                {"type": "audio", "audio": wav_path},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

    print(f"process_mm_info found: {len(audios)} audio item(s), {len(videos)} video item(s)")

    mm_data = {"video": videos}
    if audios:
        mm_data["audio"] = audios

    inputs = {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": {"use_audio_in_video": False},
    }

    out = llm.generate([inputs], sampling_params=sampling_params)[0]
    print("\n=== MODEL OUTPUT ===")
    print(out.outputs[0].text)


if __name__ == "__main__":
    main()
