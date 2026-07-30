#########################################################
# Interactive prompt loop for a fine-tuned checkpoint
#########################################################

# gpt2-small124M-lyrics-sft.pth

import argparse
import glob
import os
import time

import tiktoken
import torch

from utils import configure_device
from gpt import GPTModel, generate, text_to_token_ids, token_ids_to_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_MODEL = "gpt2-medium355M-sft.pth"
EOS_ID = 50256
NEWLINE_TOKEN_IDS = {198, 628}  # '\n', '\n\n' — must stay free to recur under repetition_penalty

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "num_layers": 12, "num_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "num_layers": 24, "num_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "num_layers": 36, "num_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "num_layers": 48, "num_heads": 25},
}
# The checkpoint tells us the size, so look the rest of the architecture up by emb_dim
configs_by_emb_dim = {config["emb_dim"]: (name, config) for name, config in model_configs.items()}


PROMPT_STYLES = {
    "lyrics": {
        "header": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request. "
        ),
        "temperature": 0.8,
        "top_k": 50,
        "max_new_tokens": 300,
        "repetition_penalty": 1.15,
    },
    "instruction": {
        "header": (
            "Below is an instruction that describes a taks. "
            "Write a response that appropriately completes the request. "
        ),
        "temperature": 0.0,
        "top_k": None,
        "max_new_tokens": 256,
        "repetition_penalty": 1.0,
    },
}

EXIT_COMMANDS = {"exit", "quit", ":q"}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a fine-tuned GPT-2 checkpoint and prompt it interactively."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"checkpoint file name in data/, or a path to one (default: {DEFAULT_MODEL})"
    )
    parser.add_argument("--temperature", type=float, help="0.0 is greedy decoding")
    parser.add_argument("--top-k", type=int, help="sample from the k most likely tokens")
    parser.add_argument("--max-new-tokens", type=int, help="response length cap in tokens")
    parser.add_argument("--repetition-penalty", type=float,
                        help="above 1.0 discourages repeating tokens (1.0 disables)")
    return parser.parse_args()

def resolve_checkpoint(name):
    path = name if os.path.isabs(name) or os.path.dirname(name) else os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        return path

    available = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, "*.pth")))
    listing = "\n  ".join(available) if available else "(none found in data/)"
    raise SystemExit(f"Checkpoint not found: {path}\n\nAvailable checkpoints:\n  {listing}")

def prompt_style(checkpoint):
    name = "lyrics" if "lyrics" in os.path.basename(checkpoint).lower() else "instruction"

    return name, dict(PROMPT_STYLES[name])

def format_prompt(header, instruction, input_text):
    prompt = f"{header}\n\n### Instruction:\n{instruction}"
    if input_text:
        prompt += f"\n\n### Input:\n{input_text}"

    return prompt + "\n\n### Response:\n"

def split_prompt(line):
    # "instruction :: input" lets the user fill the optional ### Input field
    instruction, separator, input_text = line.partition("::")
    if not separator:
        return line.strip(), ""

    return instruction.strip(), input_text.strip()

def load_model(checkpoint, device):
    try:
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    except RuntimeError as error:
        # A save that was interrupted still leaves a structurally valid zip, so the
        # missing tensors only show up here as a stream-reader failure
        if "failed locating file" not in str(error):
            raise
        size_mb = os.path.getsize(checkpoint) / 1024 ** 2
        raise SystemExit(
            f"Checkpoint is incomplete: {checkpoint} ({size_mb:.0f} MB)\n\n"
            f"It is missing most of its tensors, which happens when the save that "
            f"produced it was interrupted. The weights cannot be recovered from the "
            f"file; delete it and re-run the fine-tune script that writes it."
        ) from error

    emb_dim = state_dict["tok_emb.weight"].shape[1]
    if emb_dim not in configs_by_emb_dim:
        raise SystemExit(f"Unrecognized embedding dimension {emb_dim} in {checkpoint}")
    model_name, size_config = configs_by_emb_dim[emb_dim]

    config = {
        "vocab_size": state_dict["tok_emb.weight"].shape[0],
        "context_length": state_dict["pos_emb.weight"].shape[0],
        "drop_rate": 0.0,  # eval only, so dropout never fires
        "qkv_bias": True,
    }
    config.update(size_config)

    model = GPTModel(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config, model_name

def respond(model, tokenizer, settings, config, device, prompt):
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=settings["max_new_tokens"],
        context_size=config["context_length"],
        temperature=settings["temperature"],
        top_k=settings["top_k"],
        eos_id=EOS_ID,
        repetition_penalty=settings.get("repetition_penalty", 1.0),
        repetition_penalty_exclude=NEWLINE_TOKEN_IDS,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    return generated_text[len(prompt):].strip()

def main():
    args = parse_args()
    checkpoint = resolve_checkpoint(args.model)
    style_name, settings = prompt_style(checkpoint)

    if args.temperature is not None:
        settings["temperature"] = args.temperature
    if args.top_k is not None:
        settings["top_k"] = args.top_k
    if args.max_new_tokens is not None:
        settings["max_new_tokens"] = args.max_new_tokens
    if args.repetition_penalty is not None:
        settings["repetition_penalty"] = args.repetition_penalty

    device = configure_device(torch)
    tokenizer = tiktoken.get_encoding("gpt2")

    print(f"\nDevice: {device}")
    print(f"Loading {os.path.basename(checkpoint)}...")
    model, config, model_name = load_model(checkpoint, device)

    print(f"\nModel: {model_name}  |  prompt style: {style_name}")
    print(f"Temperature: {settings['temperature']}  |  top_k: {settings['top_k']}  "
          f"|  max new tokens: {settings['max_new_tokens']}")
    print("Type a prompt and press Enter. Use 'instruction :: input' to fill the optional")
    print(f"input field. Type {', '.join(sorted(EXIT_COMMANDS))} or press Ctrl-C to leave.\n")

    while True:
        try:
            line = input("Prompt > ")
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye.\n")
            break

        if not line.strip():
            print()
            continue

        if line.strip().lower() in EXIT_COMMANDS:
            print("\nGoodbye.\n")
            break

        instruction, input_text = split_prompt(line)
        prompt = format_prompt(settings["header"], instruction, input_text)

        start_time = time.time()
        try:
            response = respond(model, tokenizer, settings, config, device, prompt)
        except KeyboardInterrupt:
            print("\n\n[generation interrupted]\n")
            continue
        elapsed = time.time() - start_time

        print(f"\nResponse ({elapsed:.1f}s):\n")
        print(response if response else "[empty response]")
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
