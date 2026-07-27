import contextlib
import math

import torch
import numpy as np

from utils import calc_loss_batch, calc_loss_loader

class MultiHeadAttention(torch.nn.Module):
    def __init__(self,
                 d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape 

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            torch.nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["num_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = torch.nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut

        return x

class GPTModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = torch.nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = torch.nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = torch.nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["num_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = torch.nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

def generate(model, idx,
             max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None,
             repetition_penalty=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Damp tokens that already appeared so the model stops looping a hook forever.
        # Dividing only demotes a token when its logit is positive, so negatives are
        # multiplied instead (the CTRL formulation).
        if repetition_penalty != 1.0:
            for row in range(idx.shape[0]):
                seen = torch.unique(idx[row])
                scores = logits[row, seen]
                logits[row, seen] = torch.where(
                    scores > 0, scores / repetition_penalty, scores * repetition_penalty
                )

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  

        if idx_next == eos_id: 
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)
    return idx

def lr_at_step(step, total_steps, peak_lr, min_lr, warmup_steps):
    """Linear warmup to peak_lr, then cosine decay down to min_lr."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer,
                       accumulation_steps=1, peak_lr=None, min_lr=0.0,
                       warmup_frac=0.1, max_grad_norm=None):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    num_batches = len(train_loader)

    # Scheduling works in optimizer steps, not micro-batches
    total_steps = math.ceil(num_batches / accumulation_steps) * num_epochs
    warmup_steps = max(1, int(total_steps * warmup_frac))

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        optimizer.zero_grad()  # Start each epoch with a clean gradient buffer

        for i, (input_batch, target_batch) in enumerate(train_loader):
            loss = calc_loss_batch(input_batch, target_batch, model, device) / accumulation_steps
            loss.backward()  # Accumulate loss gradients into .grad
            tokens_seen += input_batch.numel()

            is_last_batch = (i + 1) == num_batches
            if (i + 1) % accumulation_steps != 0 and not is_last_batch:
                continue

            if peak_lr is not None:
                # global_step is only bumped after the update, so the step about to
                # run is global_step + 1
                lr = lr_at_step(global_step + 1, total_steps, peak_lr, min_lr, warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()  # Update model weights using the accumulated gradients
            optimizer.zero_grad()  # Reset gradients for the next effective batch
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def evaluate_model_amp(model, train_loader, val_loader, device, eval_iter,
                       val_iter=None, amp_dtype=None):
    """Copy of evaluate_model that runs under autocast and can use the whole val set.

    val_iter=None walks every validation batch, which matters once the val loss is
    what decides when to stop: a 20-batch sample of it is noisy enough to trigger
    early stopping on measurement noise alone.
    """
    autocast = (torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None else contextlib.nullcontext())
    model.eval()
    with torch.no_grad(), autocast:
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=val_iter)
    model.train()
    return train_loss, val_loss


def decay_param_groups(model, weight_decay):
    """Apply weight decay to matmul weights only, never to biases or LayerNorm gains.

    Decaying a LayerNorm scale or a bias pulls it toward zero without any
    regularizing benefit, and on a fine-tune that just erodes the pretrained
    calibration.
    """
    decay, no_decay, seen = [], [], set()
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen:
            continue  # tok_emb and out_head are the same tensor once weights are tied
        seen.add(id(param))
        (decay if param.dim() >= 2 else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train_model_finetune(model, train_loader, val_loader, optimizer, device, num_epochs,
                         eval_freq, eval_iter, start_context, tokenizer,
                         accumulation_steps=1, peak_lr=None, min_lr=0.0,
                         warmup_frac=0.05, max_grad_norm=1.0, val_iter=None,
                         use_amp=False, patience=None, min_delta=0.0):
    """Copy of train_model_simple built for fine-tuning on a small dataset.

    Three additions over the simple loop:
      * fp16 autocast + gradient scaling. Most of the win is speed - Turing runs
        fp16 matmuls at 2x. Memory moves less than it looks like it should,
        because the cross-entropy over a 50k vocab is autocast back to fp32 and
        that tensor dominates the peak;
      * the best-so-far weights are kept and restored at the end, so the returned
        model is the one at the val-loss minimum rather than whatever the last
        step produced;
      * early stopping, so the run ends on its own once val loss stops improving
        instead of needing the epoch count guessed in advance.
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    num_batches = len(train_loader)

    # Scheduling works in optimizer steps, not micro-batches
    total_steps = math.ceil(num_batches / accumulation_steps) * num_epochs
    warmup_steps = max(1, int(total_steps * warmup_frac))

    # fp16 only pays off on CUDA; on CPU/MPS this degrades to a plain fp32 run
    amp_dtype = torch.float16 if (use_amp and device.type == "cuda") else None
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is not None)

    def autocast():
        return (torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None else contextlib.nullcontext())

    best_val_loss = float("inf")
    best_state, best_step = None, -1
    evals_without_improvement = 0
    stop_early = False

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        for i, (input_batch, target_batch) in enumerate(train_loader):
            with autocast():
                loss = calc_loss_batch(input_batch, target_batch, model, device) / accumulation_steps
            scaler.scale(loss).backward()  # no-op scaling when AMP is off
            tokens_seen += input_batch.numel()

            is_last_batch = (i + 1) == num_batches
            if (i + 1) % accumulation_steps != 0 and not is_last_batch:
                continue

            if peak_lr is not None:
                # global_step is only bumped after the update, so the step about to
                # run is global_step + 1
                lr = lr_at_step(global_step + 1, total_steps, peak_lr, min_lr, warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            if max_grad_norm is not None:
                # Gradients are still scaled at this point, so undo that before
                # clipping or the norm threshold means nothing
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)  # skipped automatically if the step overflowed
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model_amp(
                    model, train_loader, val_loader, device, eval_iter,
                    val_iter=val_iter, amp_dtype=amp_dtype)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                improved = val_loss < best_val_loss - min_delta
                if improved:
                    best_val_loss, best_step = val_loss, global_step
                    # Park the snapshot on the CPU; a second copy of the model does
                    # not fit next to the optimizer state in 4 GB of VRAM
                    best_state = {k: v.detach().to("cpu", copy=True)
                                  for k, v in model.state_dict().items()}
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                      f"{' *' if improved else ''}")

                if patience is not None and evals_without_improvement >= patience:
                    print(f"Early stop: no val improvement for {patience} evaluations "
                          f"(best {best_val_loss:.3f} at step {best_step}).")
                    stop_early = True
                    break

        generate_and_print_sample(model, tokenizer, device, start_context)

        if stop_early:
            break

    if best_state is not None:
        # Whatever the last step left behind is past the val minimum; roll back
        model.load_state_dict(best_state)
        print(f"Restored best weights from step {best_step} (val loss {best_val_loss:.3f}).")

    return train_losses, val_losses, track_tokens_seen, best_val_loss

def generate_and_print_sample(model, tokenizer, device, start_context,
                              temperature=0.9, top_k=50, repetition_penalty=1.15):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size,
            temperature=temperature, top_k=top_k, eos_id=50256,
            repetition_penalty=repetition_penalty
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) 
    return tokenizer.decode(flat.tolist())

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

