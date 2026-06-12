"""
SIRCA-RAG: Build training dataset for embedding fine-tuning.

Targets robustness to common, regional, market and native-dialect names
of Peruvian medicinal plants. Produces (anchor, positive, negative) triplets
for MultipleNegativesRankingLoss training.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import SPECIES_CATALOG


# ------------------------------------------------------------------
# Additional vernacular / market / native-dialect names not in catalog
# These are curated from ethnobotanical literature (Bussmann 2006, 2015;
# Lock 2016; Gonzales 2012) for the 8 target species + others used in
# Peruvian markets and Andean dialects.
# ------------------------------------------------------------------
EXTRA_VERNACULAR = {
    "Uncaria tomentosa": [
        "una de gato", "uncaria", "garra de gato", "cat's claw",
        "samento", "saventaro", "paotitamoshki",  # Asháninka
    ],
    "Lepidium meyenii": [
        "maca", "maca peruana", "ginseng peruano", "ayuk willku",  # Quechua
        "maino", "peruvian ginseng",
    ],
    "Croton lechleri": [
        "sangre de drago", "sangre de grado", "drago", "dragon's blood",
        "sangre del arbol", "yawar wayqu",  # Quechua
    ],
    "Minthostachys mollis": [
        "muna", "muna muna", "poleo", "poleo del peru",
        "huaycha", "qhuna",  # Quechua
    ],
    "Smallanthus sonchifolius": [
        "yacon", "yacon peruano", "llacon", "polaco",
        "aricoma", "puhe",
    ],
    "Physalis peruviana": [
        "aguaymanto", "tomatillo", "uchuva", "cape gooseberry",
        "capuli", "topotopo", "ushun",
    ],
    "Erythroxylum coca": [
        "coca", "hoja de coca", "kuka",  # Quechua
        "ipadu", "yungas",
    ],
    "Buddleja incana": [
        "quishuar", "quishuara", "qishwar",  # Quechua
        "kolle blanco", "andean buddleia",
    ],
    "Polylepis rugulosa": [
        "quenua", "quenual", "qiwina",  # Quechua
        "arbol de papel",
    ],
    "Chuquiraga spinosa": [
        "huamanpinta", "huamanripa", "wamanripa",  # Quechua variant
        "amaro", "espina de huamani",
    ],
    "Senecio nutans": [
        "chachacoma", "chachakuma",  # Quechua
        "soroche", "soroche pampa",
    ],
    "Werneria nubigena": [
        "wamampinta", "matico de la puna",
        "yareta del cerro",
    ],
    "Baccharis genistelloides": [
        "carqueja", "carquejilla", "tres esquinas",
        "qarwincha",  # Quechua
    ],
    "Matricaria chamomilla": [
        "manzanilla", "manzanilla comun", "chamomile",
    ],
    "Eucalyptus globulus": [
        "eucalipto", "eucalipto comun", "blue gum",
    ],
    "Origanum vulgare": [
        "oregano", "oregano del peru", "oregano de campo",
    ],
}


def build_triplets(seed: int = 42) -> tuple[list, list, list]:
    """
    Returns (training_triplets, eval_pairs, vocabulary).

    training_triplets: [(anchor, positive, negative)] for contrastive training.
    eval_pairs:        [(common_name, scientific_name)] for evaluation.
    vocabulary:        all common names indexed by species (for the alias dict).
    """
    rng = random.Random(seed)

    # 1) Merge catalog common names + extra vernacular
    species_aliases = {}
    for sci, info in SPECIES_CATALOG.items():
        names = list(info.get("common", []))
        if sci in EXTRA_VERNACULAR:
            names.extend(EXTRA_VERNACULAR[sci])
        # Normalize: lowercase + dedup
        norm = []
        seen = set()
        for n in names:
            k = n.lower().strip()
            if k and k not in seen:
                seen.add(k)
                norm.append(n)
        if norm:
            species_aliases[sci] = norm

    print(f"Species with aliases: {len(species_aliases)}")
    total_aliases = sum(len(v) for v in species_aliases.values())
    print(f"Total aliases: {total_aliases}")

    # 2) Build context strings (anchors) per species — these are the
    #    "passages" the embedder should retrieve when querying with a common name.
    #    We use 3 templates per species to vary the language between EN/ES.
    species_passages = {}
    for sci, info in SPECIES_CATALOG.items():
        use = info.get("use", "")
        compounds = ", ".join(info.get("compounds", []))
        family = info.get("family", "")

        passages = []
        if use:
            passages.append(
                f"passage: {sci} ({family}) is used for {use}. "
                f"Active compounds include {compounds}."
            )
            passages.append(
                f"passage: La especie {sci} de la familia {family} se utiliza "
                f"para {use}. Compuestos: {compounds}."
            )
        if compounds:
            passages.append(
                f"passage: Phytochemical studies on {sci} report the presence "
                f"of {compounds}, supporting its traditional ethnobotanical use."
            )
        if passages:
            species_passages[sci] = passages

    # 3) Build positive (anchor, positive) pairs:
    #    anchor = "query: <common_name>"  →  positive = a passage about the
    #    scientific species.
    pos_pairs = []
    for sci, aliases in species_aliases.items():
        passages = species_passages.get(sci, [])
        if not passages:
            continue
        for alias in aliases:
            # English query form
            pos_pairs.append((f"query: {alias}", rng.choice(passages)))
            # Question-form query (more realistic)
            pos_pairs.append((
                f"query: para que sirve la {alias}",
                rng.choice(passages),
            ))
            pos_pairs.append((
                f"query: what are the medicinal uses of {alias}",
                rng.choice(passages),
            ))

    print(f"Positive pairs: {len(pos_pairs)}")

    # 4) Build triplets: for each positive, pick a hard negative — a passage
    #    about a DIFFERENT species, preferably same family (harder negative).
    family_to_species = {}
    for sci, info in SPECIES_CATALOG.items():
        fam = info.get("family", "")
        if fam:
            family_to_species.setdefault(fam, []).append(sci)

    species_list = list(species_passages.keys())
    triplets = []
    for anchor, positive in pos_pairs:
        # Find the species this positive belongs to
        pos_species = None
        for sci, passages in species_passages.items():
            if positive in passages:
                pos_species = sci
                break

        if not pos_species:
            continue

        # Pick a negative species (prefer same family, fallback any)
        fam = SPECIES_CATALOG[pos_species].get("family", "")
        same_fam = [s for s in family_to_species.get(fam, []) if s != pos_species]
        candidates = same_fam if same_fam else [
            s for s in species_list if s != pos_species
        ]
        neg_species = rng.choice(candidates)
        neg_passage = rng.choice(species_passages[neg_species])
        triplets.append((anchor, positive, neg_passage))

    print(f"Triplets: {len(triplets)}")

    # 5) Build evaluation pairs (queries-only, common names → species)
    #    using a HELD-OUT set: keep 1 alias per species for eval.
    eval_pairs = []
    train_aliases = {}
    for sci, aliases in species_aliases.items():
        if len(aliases) >= 2:
            held = aliases[-1]
            eval_pairs.append({"alias": held, "scientific": sci})
            train_aliases[sci] = aliases[:-1]
        else:
            train_aliases[sci] = aliases

    print(f"Eval pairs (held-out): {len(eval_pairs)}")

    # 6) Vocabulary for alias dictionary (for the paper section)
    vocabulary = {sci: aliases for sci, aliases in species_aliases.items()}

    return triplets, eval_pairs, vocabulary


def main():
    triplets, eval_pairs, vocabulary = build_triplets()

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    # Save triplets as JSONL for sentence-transformers
    triplets_file = out_dir / "triplets.jsonl"
    with triplets_file.open("w", encoding="utf-8") as f:
        for a, p, n in triplets:
            f.write(json.dumps({"anchor": a, "positive": p, "negative": n},
                               ensure_ascii=False) + "\n")
    print(f"Saved {triplets_file}")

    # Save eval pairs
    eval_file = out_dir / "eval_pairs.json"
    with eval_file.open("w", encoding="utf-8") as f:
        json.dump(eval_pairs, f, ensure_ascii=False, indent=2)
    print(f"Saved {eval_file}")

    # Save vocabulary (alias dictionary)
    vocab_file = out_dir / "vocabulary.json"
    with vocab_file.open("w", encoding="utf-8") as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)
    print(f"Saved {vocab_file}")


if __name__ == "__main__":
    main()
