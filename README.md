# OmniGuard: Unified Omni-Modal Guardrails with Deliberate Reasoning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-yellow.svg)](https://huggingface.co/anonymous-omniguard)

OmniGuard introduces the first omni-modal guardrail framework designed to address the safety challenges posed by Omni-modal Large Language Models (OLLMs) that process text, images, videos, and audio. Unlike prior unimodal approaches that rely on binary safeguarding, OmniGuard delivers deliberate, structured safety reasoning across all modalities. Built atop the Qwen2.5-Omni architecture, OmniGuard demonstrates strong effectiveness and generalization on 15 diverse safety benchmarks, offering unified, robust policy enforcement and risk mitigation for next-generation multimodal AI systems.

## 🤗 Pre-trained Models

We provide two pre-trained OmniGuard models on HuggingFace:

- **[OmniGuard-7B](https://huggingface.co/anonymous-omniguard/OmniGuard-7B)** - 7 billion parameter model for high-accuracy safety evaluation
- **[OmniGuard-3B](https://huggingface.co/anonymous-omniguard/OmniGuard-3B)** – 3 billion parameter model for resource-constrained environments

Both models support multimodal inputs (text, image, audio, video) .

## Features

- **Multimodal Support**: Evaluate safety across text, images, audio, and video
- **LLaMA Guard Integration**: Uses LLaMA Guard 3 safety assessment framework
- **Extensive Dataset Compatibility**: Supports a wide range of unimodal and cross-modal datasets across text, image, audio, and video domains
- **Flexible Architecture**: Easy to extend with custom datasets and evaluators
- **Production Ready**: Clean, well-documented code suitable for research and deployment

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for model inference)
- PyTorch 2.0 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/anonymous-2654a/neurips-2026-sub.git
cd neurips-2026-sub
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Qwen2.5-Omni dependencies:
```bash
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
pip install accelerate
```

4. Download pre-trained models (optional):
```bash
# OmniGuard-7B 
huggingface-cli download anonymous-omniguard/OmniGuard-7B

# OmniGuard-3B 
huggingface-cli download anonymous-omniguard/OmniGuard-3B
```


## Quick Start

### Simple Inference with main.py

The easiest way to run inference is using the main script:

**Inference on BeaverTails Dataset:**
```bash
# Using OmniGuard-7B
python main.py --dataset beavertails --model-path anonymous-omniguard/OmniGuard-7B --num-samples 100

# Using OmniGuard-3B 
python main.py --dataset beavertails --model-path anonymous-omniguard/OmniGuard-3B --num-samples 100
```

**Inference on VLGuard Dataset (with images):**
```bash
# Using OmniGuard-7B
python main.py --dataset vlguard --model-path anonymous-omniguard/OmniGuard-7B --num-samples 100

# Using OmniGuard-3B (faster)
python main.py --dataset vlguard --model-path anonymous-omniguard/OmniGuard-3B --num-samples 100
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) by Alibaba Cloud
- Uses [LLaMA Guard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/) safety assessment framework

