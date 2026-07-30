"""
Sample review text from a trained checkpoint.

Usage:
    python src/generate.py --title "The Dark Knight" --rating 5.0
    python src/generate.py --title "Cats" --rating 0.5 --temperature 0.9 --top-k 40
"""

import argparse
import os

import torch
import torch.nn.functional as F

from model import GPTConfig, ReviewGenerator
from tokenizer import Tokenizer


def generate(model, tokenizer, prefix, device, max_new_tokens, temperature, top_k):
    eos_id = tokenizer.special_tokens["<|endoftext|>"]
    tokens = tokenizer.encode(prefix) or [eos_id]
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if idx.size(1) >= model.position_embedding.num_embeddings:
                break
            logits, _ = model(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == eos_id:
                break

    return tokenizer.decode(idx[0].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default="", help="Movie title to condition on")
    parser.add_argument("--rating", default="", help="Rating to condition on, e.g. 4.5")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "ckpt_best.pth"),
    )
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(data_dir, "tokenizer.model"))

    config = GPTConfig(vocab_size=8013, n_embeddings=128, n_heads=4, n_layers=4, block_size=512, dropout=0.2)
    model = ReviewGenerator(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"loaded checkpoint from epoch {checkpoint['epoch'] + 1}, best test loss {checkpoint['best_test_loss']:.4f}")

    prefix = ""
    if args.title:
        prefix += f"<|startoftitle|>{args.title}<|endoftitle|>"
    if args.rating:
        prefix += f"<|rating{args.rating}|>"

    text = generate(model, tokenizer, prefix, device, args.max_new_tokens, args.temperature, args.top_k)
    print("\n" + text)


if __name__ == "__main__":
    main()
