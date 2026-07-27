
import json
import os
import random
import re
import time

import tiktoken
import torch

from torch.utils.data import Dataset, DataLoader
from utils import configure_device, download_and_load_gpt2, calc_loss_loader, plot_losses
from gpt import GPTModel, load_weights_into_gpt, train_model_finetune, decay_param_groups
from functools import partial

RESPONSE_HEADER = "\n\n### Response:\n"

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request. "
        f"\n\n### Instruction:\n{entry["instruction"]}"
    )
    input_text = f"\n\n### Input:\n{entry["input"]}" if entry["input"] else ""

    return instruction_text + input_text

def collate(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

def collate_with_prompt_mask(batch, pad_token_id=50256, ignore_index=-100,
                             allowed_max_length=None, device="cpu"):
    """Copy of collate that also drops the instruction prompt out of the loss.

    Every sample opens with the same 40-odd token header, which is ~18% of the
    tokens in this dataset and is trivially predictable. Training on it spends
    gradient on copying boilerplate and, worse, makes the reported loss a blend of
    "reproduces the template" and "writes lyrics" - so the number keeps dropping
    while the part we care about stalls. Masking it means every token in the loss
    is a lyric token.

    Takes (token_ids, prompt_length) pairs from MaskedInstructionDataset rather
    than bare token lists.
    """
    batch_max_length = max(len(item) + 1 for item, _ in batch)
    inputs_lst, targets_lst = [], []

    for item, prompt_length in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # targets[i] is the token that follows inputs[i], so the first response
        # token sits at target index prompt_length - 1; everything before it is
        # the model predicting the prompt from the prompt
        targets[:prompt_length - 1] = ignore_index

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

def load_lyrics_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry["output"]}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

class MaskedInstructionDataset(Dataset):
    """Copy of InstructionDataset that also reports where the response starts.

    The prompt and the response are encoded separately and concatenated, so the
    boundary is exact rather than inferred - and it matches inference, where the
    prompt is the only thing tokenized before generation takes over.
    """
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        self.prompt_lengths = []

        for entry in data:
            prompt_ids = tokenizer.encode(format_input(entry) + RESPONSE_HEADER)
            response_ids = tokenizer.encode(entry["output"])
            self.encoded_texts.append(prompt_ids + response_ids)
            self.prompt_lengths.append(len(prompt_ids))

    def __getitem__(self, index):
        return self.encoded_texts[index], self.prompt_lengths[index]

    def __len__(self):
        return len(self.data)

file_path = "data/instruction-data-lyrics-large.json"

# Cutting the training set was the wrong lever against overfitting: less data
# overfits sooner, and the 0.4 subset left only ~49 optimizer steps in an epoch,
# which is why the loss was still falling when the run ended. The full pool plus
# early stopping (below) is the fix - stop at the val minimum instead of trying to
# guess an epoch budget that never reaches it.
train_fraction = 1.0  # < 1.0 only for quick smoke runs
split_seed = 123      # fixed, so the val/test sets stay identical across runs and losses stay comparable
subset_seed = None    # None draws a new random training subset every run; set an int to reproduce one

data = load_lyrics_data(file_path)
random.Random(split_seed).shuffle(data)  # the file is grouped by artist, so split off a shuffled copy

train_portion = int(len(data) * 0.85)  # 85% for training
val_portion = int(len(data) * 0.10)    # 10% for validation - it decides when to stop, so it needs the samples
test_portion = len(data) - train_portion - val_portion  # Remaining 5% for testing

train_pool = data[:train_portion]
val_data = data[train_portion:train_portion + val_portion]
test_data = data[train_portion + val_portion:]

subset_size = max(1, int(len(train_pool) * train_fraction))
train_data = random.Random(subset_seed).sample(train_pool, subset_size)
print(f"Training on {len(train_data)} of {len(train_pool)} training samples "
      f"(val: {len(val_data)}, test: {len(test_data)})")

num_workers = 0         # must stay 0: collate moves tensors to CUDA, which workers can't do
use_amp = True          # fp16 autocast; the GTX 1650 runs fp16 at 2x fp32, so this is mostly a speed win
batch_size = 2          # micro-batch: what has to fit in VRAM. Measured peaks at 400 tokens on this card:
                        # bs=2 3.3 GB, bs=4 4.7 GB, against ~3.1 GB free once the desktop has its share.
                        # Anything over that does not crash on Windows, it silently pages to system RAM
                        # and crawls - so raise this only after checking nvidia-smi.
accumulation_steps = 4  # batch_size * accumulation_steps = effective batch size of 8
eval_batch_size = 2     # no_grad frees the activations but not the fp32 cross-entropy over 50k logits,
                        # which is the peak here: bs=4 evaluation measured 3.9 GB, worse than training
max_output_length = 400 # longest sample is 386 tokens, so no response gets cut off mid-line
num_epochs = 4          # the cosine decay is stretched across this, so changing it also changes how fast
                        # the LR anneals; early stopping is expected to end the run before epoch 4

# Fixed 5e-5 for ~50 steps barely moved the weights. Warmup into a higher peak and
# a cosine decay covers far more ground in the same budget without blowing up the
# pretrained weights early, and gradient clipping keeps a bad batch from undoing it.
peak_lr = 1e-4
min_lr = 1e-5
warmup_frac = 0.05
max_grad_norm = 1.0
weight_decay = 0.1

# The val loss now drives the run: measured over the whole val set, and once it has
# not improved for `patience` checks the loop stops and rolls back to the best step.
eval_freq = 30          # optimizer steps between checks (~4 per epoch); each check costs ~80 forward passes
eval_iter = 10          # train-loss estimate only; val always uses every batch
patience = 6            # ~1.5 epochs of no improvement before the run gives up

device = configure_device(torch)
tokenizer = tiktoken.get_encoding("gpt2")
customized_collate = partial(collate_with_prompt_mask, device=device,
                             allowed_max_length=max_output_length)
print("Device:", device)

CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,  # must match the pretrained GPT-2 positional embeddings
    "drop_rate": 0.1,
    "qkv_bias": True,
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "num_layers": 12, "num_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "num_layers": 24, "num_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "num_layers": 36, "num_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "num_layers": 48, "num_heads": 25},
}
CHOOSE_MODEL = "gpt2-small (124M)"
CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
model = GPTModel(CONFIG)

settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
load_weights_into_gpt(model, params)
model.out_head.weight = model.tok_emb.weight

model.eval()

# Set up data loaders
torch.manual_seed(123)
train_dataset = MaskedInstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = MaskedInstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=eval_batch_size,
    collate_fn=customized_collate,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = MaskedInstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=eval_batch_size,
    collate_fn=customized_collate,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

model.to(device)

# Baselines before any training. These are response-only now, so they sit higher
# than the numbers from runs that also scored the prompt - the two are not comparable.
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

# Training process
start_time = time.time()
optimizer = torch.optim.AdamW(decay_param_groups(model, weight_decay), lr=peak_lr)

train_losses, val_losses, tokens_seen, best_val_loss = train_model_finetune(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
    val_iter=None,  # the full val set, so early stopping is not reacting to sampling noise
    # The sample has to be prompted the same way inference prompts it, response
    # header included, or the model spends its 50 tokens writing the header itself
    start_context=format_input(val_data[1]) + RESPONSE_HEADER, tokenizer=tokenizer,
    accumulation_steps=accumulation_steps,
    peak_lr=peak_lr, min_lr=min_lr, warmup_frac=warmup_frac, max_grad_norm=max_grad_norm,
    use_amp=use_amp, patience=patience,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes. "
      f"Best val loss: {best_val_loss:.3f}")

# The val split picked the checkpoint, so it is no longer an unbiased score. The
# test split never touched a stopping decision, which makes this the honest number.
model.eval()
with torch.no_grad():
    test_loss = calc_loss_loader(test_loader, model, device)
print(f"Held-out test loss: {test_loss:.3f}")

# Losses are recorded per evaluation, so map them back onto the epochs actually run
last_step = max(0, len(train_losses) - 1) * eval_freq  # the first evaluation lands on step 0
epochs_run = last_step * accumulation_steps / max(1, len(train_loader))
epochs_tensor = torch.linspace(0, epochs_run, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)  # blocks until the window is closed

# Save the model
os.makedirs("data", exist_ok=True)
file_name = os.path.join("data", f"{re.sub(r"[ ()]", "", CHOOSE_MODEL)}-lyrics-sft.pth")
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")
