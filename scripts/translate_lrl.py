"""
Translate PolyRefuse data to new low-resource languages (Swahili, Amharic).

Generates 6 files per language in PolyRefuse/:
  harmful_train_translated_{lang}.json   (260 items, field: instruction, category)
  harmful_val_translated_{lang}.json     (39 items,  field: instruction, category)
  harmful_test_translated_{lang}.json    (572 items, fields: instruction, instruction_translated, category)
  harmless_train_translated_{lang}.json  (200 items, field: instruction, category)
  harmless_val_translated_{lang}.json    (200 items, field: instruction, category)
  harmless_test_translated_{lang}.json   (500 items, fields: instruction, instruction_translated, category)

Usage:
  cd Multilingual-Refusal/
  python scripts/translate_lrl.py
"""

import json
import time
from pathlib import Path
from tqdm import tqdm
from deep_translator import GoogleTranslator

PROJECT_ROOT = Path(__file__).parent.parent
POLYREFUSE_DIR = Path.home() / "Downloads" / "experiment_thesis" / "ployrefuse_Enhanced"
SPLITS_DIR = PROJECT_ROOT / "dataset" / "splits"

# Source files for test splits (read from original PolyRefuse, not the output dir)
_ORIGINAL_POLYREFUSE = PROJECT_ROOT / "PolyRefuse"

TARGET_LANGS = ["sw", "am"]  # Swahili, Amharic

SOURCES = {
    # (data_type, split) -> (source_file, n_items, has_original_field)
    ("harmful",  "train"): (SPLITS_DIR / "harmful_train.json",                           260, False),
    ("harmful",  "val"):   (SPLITS_DIR / "harmful_val.json",                              39,  False),
    ("harmful",  "test"):  (_ORIGINAL_POLYREFUSE / "harmful_test_translated_en.json",      572, True),
    ("harmless", "train"): (SPLITS_DIR / "harmless_train_200_sampled.json",               200, False),
    ("harmless", "val"):   (SPLITS_DIR / "harmless_val_200_sampled.json",                 200, False),
    ("harmless", "test"):  (_ORIGINAL_POLYREFUSE / "harmless_test_translated_en.json",     500, True),
}


def translate_file(source_path, lang, has_original, expected_n):
    translator = GoogleTranslator(source="en", target=lang)
    with open(source_path) as f:
        items = json.load(f)

    results = []
    failed = 0
    for item in tqdm(items, desc=f"{source_path.name} -> {lang}"):
        src_text = item["instruction"]
        try:
            translated = translator.translate(src_text)
            time.sleep(0.05)  # gentle rate limit
        except Exception as e:
            print(f"\n  [WARN] translation failed: {e}")
            translated = src_text  # fall back to English
            failed += 1

        if has_original:
            results.append({
                "instruction": src_text,
                "instruction_translated": translated,
                "category": item.get("category"),
            })
        else:
            results.append({
                "instruction": translated,
                "category": item.get("category"),
            })

    print(f"  done: {len(results)} items ({failed} fallbacks)")
    return results


def main():
    POLYREFUSE_DIR.mkdir(exist_ok=True)

    for lang in TARGET_LANGS:
        print(f"\n{'='*50}")
        print(f"Language: {lang}")
        print(f"{'='*50}")

        for (dtype, split), (src, expected_n, has_original) in SOURCES.items():
            out_path = POLYREFUSE_DIR / f"{dtype}_{split}_translated_{lang}.json"

            if out_path.exists():
                print(f"  [SKIP] {out_path.name} already exists")
                continue

            print(f"\n  {dtype}_{split} ({expected_n} items)")
            results = translate_file(src, lang, has_original, expected_n)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  saved -> {out_path.name}")


if __name__ == "__main__":
    main()
