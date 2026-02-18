"""
Pytest configuration and shared fixtures for Ottoman NER tests.
"""

import json
import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ============================================================================
# Mock heavy dependencies before importing ottoman_ner
# ============================================================================

# Mock torch
torch_mock = MagicMock()
torch_mock.cuda = MagicMock()
torch_mock.cuda.is_available = MagicMock(return_value=False)
torch_mock.tensor = lambda x, **kwargs: x
# Mock no_grad context manager properly
class NoGradContext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
torch_mock.no_grad = MagicMock(return_value=NoGradContext())

# Create mock tensor class that behaves like numpy array
class MockTensor:
    def __init__(self, data):
        if isinstance(data, list):
            self.data = np.array(data)
        else:
            self.data = data
    
    def __getitem__(self, idx):
        result = self.data[idx]
        if isinstance(result, np.ndarray):
            return MockTensor(result)
        return result
    
    def item(self):
        if isinstance(self.data, np.ndarray):
            return int(self.data.flat[0])
        return self.data
    
    def tolist(self):
        if isinstance(self.data, np.ndarray):
            return self.data.tolist()
        return self.data

def mock_argmax(data, dim=None):
    if isinstance(data, MockTensor):
        arr = data.data
    else:
        arr = np.array(data)
    if dim is not None:
        result = np.argmax(arr, axis=dim)
        return MockTensor(result)
    return np.argmax(arr)

def mock_softmax(data, dim=None):
    if isinstance(data, MockTensor):
        arr = data.data
    else:
        arr = np.array(data)
    if dim is not None:
        # Simple softmax implementation
        exp_arr = np.exp(arr - np.max(arr, axis=dim, keepdims=True))
        result = exp_arr / np.sum(exp_arr, axis=dim, keepdims=True)
        return MockTensor(result)
    exp_arr = np.exp(arr - np.max(arr))
    return exp_arr / np.sum(exp_arr)

torch_mock.argmax = mock_argmax
torch_mock.nn = MagicMock()
torch_mock.nn.functional = MagicMock()
torch_mock.nn.functional.softmax = mock_softmax
sys.modules['torch'] = torch_mock

# Mock transformers
autotokenizer_mock = MagicMock()
automodel_mock = MagicMock()
training_args_mock = MagicMock()
trainer_mock = MagicMock()
data_collator_mock = MagicMock()

transformers_mock = MagicMock()
transformers_mock.AutoTokenizer = autotokenizer_mock
transformers_mock.AutoModelForTokenClassification = automodel_mock
transformers_mock.TrainingArguments = training_args_mock
transformers_mock.Trainer = trainer_mock
transformers_mock.DataCollatorForTokenClassification = data_collator_mock
sys.modules['transformers'] = transformers_mock

# Mock datasets - create a proper Dataset class
class MockDataset:
    def __init__(self, data):
        self.data = data
        self.features = list(data.keys()) if data else []
    
    def __len__(self):
        if not self.data:
            return 0
        return len(next(iter(self.data.values())))
    
    def __getitem__(self, idx):
        # Support both integer indexing and string column access
        if isinstance(idx, str):
            return self.data.get(idx, [])
        return {k: v[idx] for k, v in self.data.items()}
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def map(self, func, batched=False):
        # Simple mock that just returns self
        return self
    
    @classmethod
    def from_dict(cls, data):
        return cls(data)

# Create module-like mock
datasets_module = type(sys)('datasets')
datasets_module.Dataset = MockDataset
sys.modules['datasets'] = datasets_module

# Mock seqeval
seqeval_metrics_mock = MagicMock()
seqeval_metrics_mock.classification_report = MagicMock(return_value={
    "PER": {"precision": 0.9, "recall": 0.85, "f1-score": 0.87},
    "overall_precision": 0.90,
    "overall_recall": 0.85,
    "overall_f1": 0.87,
})
seqeval_metrics_mock.f1_score = MagicMock(return_value=0.85)
seqeval_metrics_mock.precision_score = MagicMock(return_value=0.90)
seqeval_metrics_mock.recall_score = MagicMock(return_value=0.80)

seqeval_mock = MagicMock()
seqeval_mock.metrics = seqeval_metrics_mock
sys.modules['seqeval'] = seqeval_mock
sys.modules['seqeval.metrics'] = seqeval_metrics_mock

# Now we can import ottoman_ner
from ottoman_ner import OttomanNER

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_conll_data():
    """Sample CoNLL format data for testing."""
    return """Sultan B-PER
Abdülhamid I-PER
İstanbul B-LOC
gitti O
. O

Osmanlı B-ORG
Devleti I-ORG
kuruldu O
. O

"""


@pytest.fixture
def sample_conll_file(temp_dir, sample_conll_data):
    """Create a sample CoNLL file for testing."""
    file_path = temp_dir / "sample.conll"
    file_path.write_text(sample_conll_data, encoding="utf-8")
    return str(file_path)


@pytest.fixture
def empty_conll_file(temp_dir):
    """Create an empty CoNLL file for testing."""
    file_path = temp_dir / "empty.conll"
    file_path.write_text("", encoding="utf-8")
    return str(file_path)


@pytest.fixture
def malformed_conll_file(temp_dir):
    """Create a malformed CoNLL file for testing."""
    file_path = temp_dir / "malformed.conll"
    file_path.write_text("""Sultan

Abdülhamid I-PER extra
""", encoding="utf-8")
    return str(file_path)


@pytest.fixture
def mock_model_config():
    """Mock model configuration."""
    config = MagicMock()
    config.id2label = {
        "0": "O",
        "1": "B-PER",
        "2": "I-PER",
        "3": "B-LOC",
        "4": "I-LOC",
        "5": "B-ORG",
        "6": "I-ORG",
    }
    config.label2id = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-LOC": 3,
        "I-LOC": 4,
        "B-ORG": 5,
        "I-ORG": 6,
    }
    return config


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing that returns an object with word_ids method."""
    
    class MockTensor:
        """Mock tensor that supports .to() method."""
        def __init__(self, data):
            self._data = data
        
        def to(self, device):
            return self
        
        def __iter__(self):
            return iter(self._data)
        
        def __getitem__(self, idx):
            return self._data[idx]
    
    class TokenizerOutput:
        """Class to mimic tokenizer output with word_ids method."""
        def __init__(self, data, word_ids_list):
            self._data = data
            self._word_ids = word_ids_list
            # Wrap lists in MockTensor for .to() support
            for key, value in self._data.items():
                if isinstance(value, list):
                    self._data[key] = MockTensor(value)
        
        def __getitem__(self, key):
            return self._data[key]
        
        def word_ids(self, batch_index=0):
            return self._word_ids[batch_index] if batch_index < len(self._word_ids) else self._word_ids[0]
        
        def items(self):
            return self._data.items()
        
        def keys(self):
            return self._data.keys()
        
        def values(self):
            return self._data.values()
    
    tokenizer = MagicMock()
    
    def mock_call(*args, **kwargs):
        # Return a TokenizerOutput with word_ids method
        return TokenizerOutput(
            {
                "input_ids": [[101, 1234, 5678, 9012, 102]],
                "attention_mask": [[1, 1, 1, 1, 1]],
            },
            [[None, 0, 0, 1, None]]  # word_ids for each token
        )
    
    tokenizer.side_effect = mock_call
    tokenizer.is_fast = True
    
    return tokenizer


@pytest.fixture
def mock_model_output():
    """Create mock model output for testing."""
    output = MagicMock()
    # Shape: (batch_size=1, seq_len=5, num_labels=7)
    # Predict B-PER, I-PER, B-LOC, O, O for tokens
    logits = np.array([
        [
            [0.1, 0.6, 0.1, 0.1, 0.05, 0.03, 0.02],  # B-PER (index 1)
            [0.1, 0.1, 0.7, 0.05, 0.02, 0.02, 0.01],  # I-PER (index 2)
            [0.1, 0.05, 0.05, 0.75, 0.02, 0.02, 0.01],  # B-LOC (index 3)
            [0.9, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01],  # O (index 0)
            [0.85, 0.03, 0.03, 0.03, 0.03, 0.02, 0.01],  # O (index 0)
        ]
    ])
    output.logits = logits
    return output


@pytest.fixture
def mock_ner_instance(mock_tokenizer, mock_model_output, mock_model_config):
    """Create a mock OttomanNER instance with mocked model and tokenizer."""
    ner = OttomanNER()
    ner.tokenizer = mock_tokenizer
    
    # Mock model
    mock_model = MagicMock()
    mock_model.config = mock_model_config
    mock_model.return_value = mock_model_output
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    
    ner.model = mock_model
    ner.device = "cpu"
    
    return ner


@pytest.fixture
def training_config(temp_dir, sample_conll_data):
    """Create a training configuration for testing."""
    # Create train and dev files
    train_file = temp_dir / "train.conll"
    train_file.write_text(sample_conll_data, encoding="utf-8")
    
    dev_file = temp_dir / "dev.conll"
    dev_file.write_text(sample_conll_data, encoding="utf-8")
    
    # Create output directory
    output_dir = temp_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    return {
        "model": {
            "model_name_or_path": "dbmdz/bert-base-turkish-cased",
        },
        "data": {
            "train_file": str(train_file),
            "dev_file": str(dev_file),
        },
        "training": {
            "output_dir": str(output_dir),
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
        }
    }


@pytest.fixture
def sample_predictions_and_labels():
    """Sample predictions and labels for metrics testing."""
    # predictions shape: (batch_size, seq_len, num_labels)
    # We'll create 2 examples with 5 tokens each, 3 labels (O, B-PER, I-PER)
    predictions = np.array([
        [
            [0.9, 0.05, 0.05],  # O
            [0.1, 0.8, 0.1],    # B-PER
            [0.1, 0.1, 0.8],    # I-PER
            [0.85, 0.1, 0.05],  # O
            [0.9, 0.05, 0.05],  # O
        ],
        [
            [0.85, 0.1, 0.05],  # O
            [0.1, 0.85, 0.05],  # B-PER
            [0.9, 0.05, 0.05],  # O
            [0.1, 0.1, 0.8],    # I-PER (wrong - should be O)
            [0.9, 0.05, 0.05],  # O
        ]
    ])
    
    # Labels: -100 for ignored tokens
    labels = np.array([
        [0, 1, 2, 0, -100],   # O, B-PER, I-PER, O, ignored
        [0, 1, 0, 0, -100],   # O, B-PER, O, O, ignored
    ])
    
    return predictions, labels


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging handlers after each test."""
    # Store original state
    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_handlers = root_logger.handlers.copy()
    
    # Reset root logger before test
    root_logger.setLevel(logging.WARNING)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    yield
    
    # Restore original state after test
    root_logger.setLevel(original_level)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    for handler in original_handlers:
        root_logger.addHandler(handler)


@pytest.fixture
def caplog_clean(caplog):
    """Provide a clean caplog fixture."""
    caplog.clear()
    return caplog
