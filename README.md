# Text-to-Image Generation Evaluation

This repo supports evaluating multiple text-to-image generation models using a common interface.

## Supported Models

- CogView4 (`cogview`)
- Flux (`flux`) 
- Janus (`janus`)
- Infinity (`infinity`)
- SANA (`sana`)
- EMU3 (`emu3`)
- Lumina (`lumina`)

## Usage

Basic usage:
```bash
python eval/generate.py \
    --task dataset/instruction.jsonl \
    --model cogview \
    --output_dir results/cogview
```

For models requiring additional parameters:
```bash
# For Infinity model
python eval/generate.py \
    --task dataset/instruction.jsonl \
    --model infinity \
    --model_path path/to/model \
    --vae_path path/to/vae \
    --output_dir results/infinity
```

## Input Format

The input file should be a JSONL file where each line contains:
```json
{
    "task_id": "unique_id",
    "messages": [
        {
            "role": "user",
            "content": "prompt text"
        }
    ]
}
```

## Output

Generated images will be saved in the specified output directory with filenames matching their task IDs (e.g., `1.png`, `2.png`, etc.).