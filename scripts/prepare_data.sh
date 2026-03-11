#!/usr/bin/env bash
set -e

MODEL="/nlp/scr/phoongkz/models/meta-llama-Llama-3.1-8B-Instruct"

FOLDERS=(
    indonesian_english
    indonesian_english_insider
    japanese_english
    japanese_english_insider
    korean_english
    korean_english_insider
    malaysian_english
    malaysian_english_insider
    nigerian_english
    nigerian_english_insider
    thai_english
    thai_english_insider
    vietnamese_english
    vietnamese_english_insider
)

for folder in "${FOLDERS[@]}"; do
    echo "Preparing: $folder"
    python main.py finetune prepare-data \
        --input-dirs "./output/$folder" \
        --approved-only \
        --topk 1 --selection-strategy perplexity --selection-model $MODEL --tensor-parallel-size 2
done

echo "Done."
