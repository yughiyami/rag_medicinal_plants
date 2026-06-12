"""
Fine-tune multilingual-e5-base on Peruvian medicinal plants vernacular pairs.
Uses MultipleNegativesRankingLoss for contrastive learning.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
)
from torch.utils.data import DataLoader


def load_triplets(path: Path) -> list[InputExample]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            examples.append(InputExample(
                texts=[t["anchor"], t["positive"], t["negative"]]
            ))
    return examples


def main():
    data_dir = Path(__file__).parent / "data"
    out_dir = Path(__file__).parent / "model"
    out_dir.mkdir(exist_ok=True)

    print("Loading base model: intfloat/multilingual-e5-base")
    model = SentenceTransformer("intfloat/multilingual-e5-base")

    print("Loading triplets...")
    examples = load_triplets(data_dir / "triplets.jsonl")
    print(f"Triplets: {len(examples)}")

    # CPU-friendly: smaller batch + single epoch.
    # No pin_memory (CPU), no auto-save (was hanging), explicit save after.
    train_dataloader = DataLoader(
        examples, shuffle=True, batch_size=8, pin_memory=False
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    print("Training (1 epoch, batch=8, NO auto-save)...", flush=True)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=int(len(train_dataloader) * 0.1),
        show_progress_bar=True,
        output_path=None,  # skip auto-save (was hanging here)
        use_amp=False,
    )
    print("Training finished. Saving model manually...", flush=True)
    model.save(str(out_dir))
    print(f"Model saved to {out_dir}", flush=True)
    print(f"Model saved to {out_dir}")


if __name__ == "__main__":
    main()
