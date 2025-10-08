# Ottoman NER

**A focused toolkit for Ottoman Turkish Named Entity Recognition**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/ottoman-ner.svg)](https://badge.fury.io/py/ottoman-ner)

---

## About

Ottoman NER is a specialized Python package for **Named Entity Recognition (NER)** in **Ottoman Turkish** texts. This package provides a clean, modern interface for training, evaluating, and using NER models specifically designed for historical Ottoman Turkish documents.

### Key Features

- **Focused NER Solution**: Dedicated solely to Ottoman Turkish named entity recognition
- **Simple API**: Single class interface for all NER operations
- **Easy Training**: Train custom models with JSON configuration
- **Built-in Evaluation**: Comprehensive evaluation metrics with seqeval
- **Fast Prediction**: Real-time entity recognition
- **CLI Interface**: Command-line tools for all operations
- **PyPI Ready**: Easy installation via pip

### Supported Entity Types

- **PER**: Person names (Sultan Abdülhamid, Ahmet Paşa)
- **LOC**: Locations (İstanbul, Rumeli, Anadolu)
- **ORG**: Organizations (Divan-ı Hümayun, Meclis-i Mebusan)
- **MISC**: Miscellaneous entities (dates, events, titles)

---

## Installation

### From PyPI (Recommended)

```bash
pip install ottoman-ner
```

### From Source

```bash
git clone https://github.com/fbkaragoz/ottoman-ner.git
cd ottoman-ner
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Install with full features (visualization, experiment tracking)
pip install -e .[full]
```

---

## Quick Start

### 1. Using Pre-trained Models

```python
from ottoman_ner import OttomanNER

# Initialize the NER system
ner = OttomanNER()

# Load a pre-trained model
ner.load_model("models_hub/ner/ottoman-ner-standard")

# Make predictions
text = "Sultan Abdülhamid İstanbul'da yaşıyordu."
entities = ner.predict(text)

for entity in entities:
    print(f"{entity['text']} -> {entity['label']} ({entity['confidence']:.2f})")
```

### 2. Training Custom Models

```python
from ottoman_ner import OttomanNER

# Initialize
ner = OttomanNER()

# Train from configuration file
results = ner.train_from_config("configs/training.json")
print(f"Training completed! F1 Score: {results['eval_f1']:.4f}")
```

### 3. Model Evaluation

```python
from ottoman_ner import OttomanNER

# Initialize and evaluate
ner = OttomanNER()
results = ner.evaluate(
    model_path="models_hub/ner/ottoman-ner-standard",
    test_file="data/test.txt"
)

print(f"F1 Score: {results['overall_f1']:.4f}")
print(f"Precision: {results['overall_precision']:.4f}")
print(f"Recall: {results['overall_recall']:.4f}")
```

---

## Command Line Interface

Ottoman NER provides a comprehensive CLI for all operations:

### Training

```bash
# Train a new model
ottoman-ner train --config configs/training.json

# Train with verbose output
ottoman-ner --verbose train --config configs/training.json
```

### Evaluation

```bash
# Evaluate a trained model
ottoman-ner eval --model-path models_hub/ner/ottoman-ner-standard --test-file data/test.txt

# Save evaluation results
ottoman-ner eval --model-path models_hub/ner/ottoman-ner-standard --test-file data/test.txt --output-dir results/
```

### Prediction

```bash
# Predict on single text
ottoman-ner predict --model-path models_hub/ner/ottoman-ner-standard --text "Sultan Abdülhamid İstanbul'da yaşıyordu"

# Predict on file
ottoman-ner predict --model-path models_hub/ner/ottoman-ner-standard --input-file input.txt --output-file predictions.json
```

---

## Configuration

Create a training configuration file in JSON format:

```json
{
  "experiment": {
    "experiment_name": "my-ottoman-ner"
  },
  "model": {
    "model_name_or_path": "dbmdz/bert-base-turkish-cased",
    "num_labels": 9
  },
  "data": {
    "train_file": "data/train.txt",
    "dev_file": "data/dev.txt",
    "test_file": "data/test.txt",
    "max_length": 512
  },
  "training": {
    "output_dir": "models/my-model",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "learning_rate": 2e-5,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "save_steps": 100,
    "load_best_model_at_end": true,
    "metric_for_best_model": "eval_f1"
  }
}
```

---

## Data Format

Ottoman NER expects CoNLL format data with BIO tagging:

```
Sultan B-PER
Abdülhamid I-PER
İstanbul B-LOC
'da O
yaşıyordu O
. O

Osmanlı B-ORG
Devleti I-ORG
'nin O
başkenti O
İstanbul B-LOC
'dur O
. O
```

---

## Project Background & Acknowledgments

This project builds upon foundational work in Ottoman Turkish NLP and represents a focused effort to provide a clean, maintainable NER solution for historical Turkish texts.

### References

- **Karagöz et al. (2024)** — *"Towards a Clean Text Corpus for Ottoman Turkish"* [ACL Anthology](https://aclanthology.org/2024.sigturk-1.6.pdf)
- **Özateş et al. (2025)** — *"Building Foundations for Natural Language Processing of Historical Turkish: Resources and Models"* [arXiv:2501.04828](https://arxiv.org/pdf/2501.04828)

### Special Thanks

Sincere gratitude to **Assoc. Prof. Şaziye Betül Özateş** and the **Boğaziçi University Computational Linguistics Lab (BUColin)** for their foundational contributions to historical Turkish NLP.

---

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- See `requirements.txt` for complete dependencies

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use Ottoman NER in your research, please cite:

```bibtex
@software{ottoman_ner_2024,
  title={Ottoman NER: A Toolkit for Ottoman Turkish Named Entity Recognition},
  author={Karagöz, Fatih Burak},
  year={2024},
  url={https://github.com/fbkaragoz/ottoman-ner},
  version={2.0.0}
}
```

---

## Related Projects

For broader Ottoman Turkish NLP research and experimental tools, see the upcoming **`ottominer`** repository (coming soon).
