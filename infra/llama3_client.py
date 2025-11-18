"""
infra/llama3_client.py

Thin wrapper around Meta-Llama-3-8B-Instruct using Hugging Face transformers.

- Loads the model in 4-bit (bitsandbytes) to fit better on RTX 4070.
- Uses device_map="auto" so it will use GPU.
- Reads model id from LLAMA3_MODEL_ID env var (optional).
- Reads HF token from HF_TOKEN env var.

Exposes:

    get_llama3_pipeline() -> (pipe, tokenizer)
"""

import os
from typing import Tuple
from dotenv import load_dotenv   # <-- NEW

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
load_dotenv()


# Default model (you must have HF access + license accepted)
DEFAULT_MODEL_ID = os.environ.get(
    "LLAMA3_MODEL_ID",
    "meta-llama/Meta-Llama-3-8B-Instruct",
)

_tokenizer = None
_model = None
_pipe = None


def get_llama3_pipeline():
    """
    Lazy-load the LLaMA 3 8B Instruct model in 4-bit and return (pipeline, tokenizer).

    Returns:
        pipe: transformers.Pipeline for text-generation
        tokenizer: transformers.AutoTokenizer
    """
    global _tokenizer, _model, _pipe

    if _pipe is not None:
        return _pipe, _tokenizer

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set.\n"
            "Create a Hugging Face access token (with model read access), then:\n"
            "  export HF_TOKEN=your_token_here   # macOS/Linux\n"
            "  setx HF_TOKEN your_token_here      # Windows PowerShell (persist)\n"
        )

    model_id = DEFAULT_MODEL_ID
    print(f"[llama3_client] Loading model (4-bit): {model_id}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    _tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
    )

    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        device_map="auto",
        quantization_config=bnb_config,
    )

    _pipe = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
    )

    return _pipe, _tokenizer


if __name__ == "__main__":
    # Tiny smoke test (no RAG, just to see if the model loads and responds)
    pipe, tok = get_llama3_pipeline()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello to Montclair students in one sentence."},
    ]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    out = pipe(
        prompt,
        max_new_tokens=32,
        do_sample=False,
        temperature=0.2,
        pad_token_id=tok.eos_token_id,
    )[0]["generated_text"]

    answer = out[len(prompt):].strip()
    print("Model reply:\n", answer)
