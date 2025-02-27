# Tuned

A framework for fine-tuning large language models (LLMs) with emphasis on generating high-quality sci-fi content using advanced techniques like QLoRA.

## Overview

Tuned provides tools and workflows for fine-tuning foundation models (such as Phi-2) on custom datasets with minimal computational resources. The project is particularly focused on enhancing model capabilities for generating science fiction narratives with complex characters, intricate plotlines, and rich world-building.

## Features

- Data preparation and preprocessing utilities for text data
- Implementation of QLoRA (Quantized Low-Rank Adaptation) for efficient fine-tuning
- Integration with Hugging Face's Transformers, PEFT, and Accelerate libraries
- Experiment tracking with Weights & Biases
- Asynchronous prompt generation capabilities
- Utilities for parsing and structuring text data from various sources

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tuned.git
cd tuned

# Create and activate a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```
tuned/
├── data/               # Training and validation datasets
├── src/                # Source code
│   ├── helpers/        # Helper functions and utilities
│   │   ├── data_prep.py  # Data preparation utilities
├── wip/                # Works in progress and experiments
│   ├── 1_paragraphs.ipynb        # Paragraph processing notebook
│   ├── finetuneowndataphi.ipynb  # Phi model fine-tuning notebook
├── output.csv          # Generated prompts and paragraphs
├── .gitignore          # Git ignore file
└── README.md           # This file
```

## Usage

### Data Preparation

```python
from src.helpers.data_prep import parse_into_paragraphs, generate_prompt_from_segments

# Parse text into paragraphs
paragraphs = parse_into_paragraphs("data/your_text_file.txt")

# Generate prompts from paragraphs
generate_prompt_from_segments(paragraphs, "output.csv", model="your_model_name")
```

### Fine-tuning

The repository includes Jupyter notebooks demonstrating fine-tuning workflows:

1. `wip/finetuneowndataphi.ipynb` - Shows how to fine-tune Phi-2 model using QLoRA
2. `wip/1_paragraphs.ipynb` - Demonstrates processing paragraphs and generating prompts

## Technical Details

### QLoRA Implementation

QLoRA combines quantization and LoRA (Low-Rank Adaptation) to significantly reduce the memory footprint during fine-tuning while maintaining performance. This enables fine-tuning of large models on consumer-grade hardware.

Key parameters:
- 8-bit quantization for base model
- Rank: 32
- Alpha: 16
- Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, fc1, fc2

### Data Processing Pipeline

1. Text parsing and cleaning using ftfy
2. Paragraph or sentence segmentation
3. Prompt engineering using LLM-based techniques
4. Train/test split preparation
5. Format conversion for model ingestion

### Models Supported

- Phi-2
- Other Hugging Face compatible models

## Examples

### Generating Prompts from Text

```python
import pandas as pd
from src.helpers.data_prep import get_preparation_prompt, generate_text

paragraph = "Your sci-fi paragraph here"
prompt = get_preparation_prompt(paragraph)
response = generate_text(prompt, model="your_model")

# Create DataFrame and save to CSV
df = pd.DataFrame({'Original Paragraph': [paragraph], 'Prompt': [response]})
df.to_csv('output.csv', index=False)
```

### Training Setup

```python
# Sample training configuration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    weight_decay=0.001,
    warmup_steps=100,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is available under [License Name]. Note that data files may have separate licensing terms, especially those derived from Project Gutenberg or other sources with specific usage terms.

## Acknowledgments

- Hugging Face for their transformers, PEFT, and Accelerate libraries
- Project Gutenberg for text data resources
- Brev.dev for providing computational resources