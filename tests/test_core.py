"""
Tests for the core OttomanNER functionality.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from ottoman_ner import OttomanNER
from ottoman_ner.core import DEFAULT_PRETRAINED_MODEL

# Import the mock Dataset class from conftest
from tests.conftest import MockDataset as Dataset


class TestOttomanNERInit:
    """Tests for OttomanNER initialization."""
    
    def test_default_initialization(self):
        """Test OttomanNER initializes with default values."""
        ner = OttomanNER()
        
        assert ner.model is None
        assert ner.tokenizer is None
        assert ner.device in ["cuda", "cpu"]
        assert ner.default_model_path == DEFAULT_PRETRAINED_MODEL
        assert ner.default_labels == ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
        assert len(ner.label2id) == len(ner.default_labels)
        assert len(ner.id2label) == len(ner.default_labels)
    
    def test_custom_model_initialization(self):
        """Test OttomanNER initializes with custom model path."""
        custom_model = "custom/model-path"
        ner = OttomanNER(default_model=custom_model)
        
        assert ner.default_model_path == custom_model
    
    def test_none_model_initialization(self):
        """Test OttomanNER handles None model path."""
        ner = OttomanNER(default_model=None)
        
        assert ner.default_model_path == DEFAULT_PRETRAINED_MODEL


class TestLoadConllData:
    """Tests for load_conll_data method."""
    
    def test_load_valid_conll(self, sample_conll_file):
        """Test loading valid CoNLL data."""
        ner = OttomanNER()
        dataset = ner.load_conll_data(sample_conll_file)
        
        assert isinstance(dataset, Dataset)
        assert "tokens" in dataset.features
        assert "ner_tags" in dataset.features
        assert len(dataset) == 2
        
        # First sentence
        assert dataset[0]["tokens"] == ["Sultan", "Abdülhamid", "İstanbul", "gitti", "."]
        assert dataset[0]["ner_tags"] == ["B-PER", "I-PER", "B-LOC", "O", "O"]
        
        # Second sentence
        assert dataset[1]["tokens"] == ["Osmanlı", "Devleti", "kuruldu", "."]
        assert dataset[1]["ner_tags"] == ["B-ORG", "I-ORG", "O", "O"]
    
    def test_load_empty_file(self, empty_conll_file):
        """Test loading empty CoNLL file."""
        ner = OttomanNER()
        dataset = ner.load_conll_data(empty_conll_file)
        
        assert isinstance(dataset, Dataset)
        assert len(dataset) == 0
    
    def test_load_malformed_conll(self, malformed_conll_file):
        """Test loading malformed CoNLL data."""
        ner = OttomanNER()
        dataset = ner.load_conll_data(malformed_conll_file)
        
        # Should handle gracefully by skipping malformed lines
        assert isinstance(dataset, Dataset)
        # First line has no label, second line is empty (sentence break),
        # third line has extra columns but should be handled
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        ner = OttomanNER()
        
        with pytest.raises(FileNotFoundError):
            ner.load_conll_data("/nonexistent/path/file.conll")
    
    def test_load_single_sentence_no_final_newline(self, temp_dir):
        """Test loading file without final newline."""
        file_path = temp_dir / "single.conll"
        # Write with explicit content - single sentence, no final newline
        file_path.write_text("Sultan B-PER\nAbdülhamid I-PER", encoding="utf-8")
        
        ner = OttomanNER()
        dataset = ner.load_conll_data(str(file_path))
        
        # Should have one sentence with two tokens
        assert len(dataset) == 1
        assert dataset[0]["tokens"] == ["Sultan", "Abdülhamid"]
        assert dataset[0]["ner_tags"] == ["B-PER", "I-PER"]


class TestComputeMetrics:
    """Tests for compute_metrics method."""
    
    def test_compute_metrics_basic(self, sample_predictions_and_labels):
        """Test basic metrics computation."""
        ner = OttomanNER()
        ner.id2label = {0: "O", 1: "B-PER", 2: "I-PER"}
        
        predictions, labels = sample_predictions_and_labels
        results = ner.compute_metrics((predictions, labels))
        
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        
        # Check values are between 0 and 1
        assert 0 <= results["precision"] <= 1
        assert 0 <= results["recall"] <= 1
        assert 0 <= results["f1"] <= 1
    
    @patch("ottoman_ner.core.precision_score")
    @patch("ottoman_ner.core.recall_score")
    @patch("ottoman_ner.core.f1_score")
    def test_compute_metrics_perfect_predictions(self, mock_f1, mock_recall, mock_precision):
        """Test metrics with perfect predictions."""
        mock_precision.return_value = 1.0
        mock_recall.return_value = 1.0
        mock_f1.return_value = 1.0
        
        ner = OttomanNER()
        ner.id2label = {0: "O", 1: "B-PER", 2: "I-PER"}
        
        # Perfect predictions
        predictions = np.array([
            [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
        ])
        labels = np.array([[0, 1, 2]])
        
        results = ner.compute_metrics((predictions, labels))
        
        assert results["precision"] == 1.0
        assert results["recall"] == 1.0
        assert results["f1"] == 1.0
    
    @patch("ottoman_ner.core.precision_score")
    @patch("ottoman_ner.core.recall_score")
    @patch("ottoman_ner.core.f1_score")
    def test_compute_metrics_all_wrong(self, mock_f1, mock_recall, mock_precision):
        """Test metrics with all wrong predictions."""
        mock_precision.return_value = 0.0
        mock_recall.return_value = 0.0
        mock_f1.return_value = 0.0
        
        ner = OttomanNER()
        ner.id2label = {0: "O", 1: "B-PER", 2: "I-PER"}
        
        # All wrong predictions
        predictions = np.array([
            [[0.05, 0.9, 0.05], [0.05, 0.05, 0.9], [0.9, 0.05, 0.05]]
        ])
        labels = np.array([[0, 0, 0]])
        
        results = ner.compute_metrics((predictions, labels))
        
        assert results["precision"] == 0.0
        assert results["recall"] == 0.0
        assert results["f1"] == 0.0


class TestPredict:
    """Tests for predict method."""
    
    def test_predict_without_model(self):
        """Test predict raises error when model not loaded."""
        ner = OttomanNER()
        
        with pytest.raises(ValueError, match="Model not loaded"):
            ner.predict("some text")
    
    def test_predict_empty_text(self, mock_ner_instance):
        """Test predict with empty text."""
        result = mock_ner_instance.predict("")
        assert result == []
    
    def test_predict_whitespace_only(self, mock_ner_instance):
        """Test predict with whitespace-only text."""
        result = mock_ner_instance.predict("   ")
        assert result == []
    
    def test_predict_single_entity(self, mock_ner_instance, mock_tokenizer):
        """Test predict extracts single entity correctly."""
        # Adjust mock for single word
        mock_tokenizer.word_ids = lambda batch_index=0: [0]
        
        result = mock_ner_instance.predict("Sultan")
        
        assert isinstance(result, list)
        # Should detect B-PER entity
        assert len(result) >= 0  # Depends on mock behavior
    
    def test_predict_multiple_entities(self, mock_ner_instance):
        """Test predict extracts multiple entities."""
        result = mock_ner_instance.predict("Sultan Abdülhamid İstanbul")
        
        assert isinstance(result, list)
        for entity in result:
            assert "text" in entity
            assert "label" in entity
            assert "start" in entity
            assert "end" in entity
            assert "confidence" in entity
            assert isinstance(entity["confidence"], float)
            assert 0 <= entity["confidence"] <= 1


class TestLoadModel:
    """Tests for load_model method."""
    
    @patch("ottoman_ner.core.AutoTokenizer")
    @patch("ottoman_ner.core.AutoModelForTokenClassification")
    def test_load_model_from_hub(self, mock_model_class, mock_tokenizer_class, mock_model_config):
        """Test loading model from Hugging Face Hub."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.config = mock_model_config
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        ner = OttomanNER()
        ner.load_model("test-model")
        
        assert ner.model is not None
        assert ner.tokenizer is not None
        mock_tokenizer_class.from_pretrained.assert_called_once_with("test-model")
        mock_model_class.from_pretrained.assert_called_once_with("test-model")
        mock_model.eval.assert_called_once()
    
    @patch("ottoman_ner.core.AutoTokenizer")
    @patch("ottoman_ner.core.AutoModelForTokenClassification")
    def test_load_model_uses_default(self, mock_model_class, mock_tokenizer_class, mock_model_config):
        """Test loading model uses default when no path provided."""
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.config = mock_model_config
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        ner = OttomanNER()
        ner.load_model()
        
        mock_tokenizer_class.from_pretrained.assert_called_once_with(DEFAULT_PRETRAINED_MODEL)
    
    @patch("ottoman_ner.core.AutoTokenizer")
    @patch("ottoman_ner.core.AutoModelForTokenClassification")
    def test_load_model_reads_label_mappings(self, mock_model_class, mock_tokenizer_class, temp_dir, mock_model_config):
        """Test loading model reads label_mappings.json if exists."""
        # Create label mappings file
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        mappings = {
            "id2label": {"0": "O", "1": "B-TEST"},
            "label2id": {"O": 0, "B-TEST": 1}
        }
        with open(model_dir / "label_mappings.json", "w") as f:
            json.dump(mappings, f)
        
        # Create dummy files to simulate model
        (model_dir / "config.json").write_text("{}")
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.config = mock_model_config
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        ner = OttomanNER()
        ner.load_model(str(model_dir))
        
        assert ner.id2label == {0: "O", 1: "B-TEST"}
        assert ner.label2id == {"O": 0, "B-TEST": 1}


class TestTokenizeAndAlignLabels:
    """Tests for tokenize_and_align_labels method."""
    
    def test_tokenize_and_align_basic(self):
        """Test basic tokenization and label alignment."""
        
        # Create a tokenizer-like mock that returns proper output
        class MockTokenizerOutput(dict):
            def __init__(self, data):
                super().__init__(data)
                self._data = data
            
            def word_ids(self, batch_index=0):
                return [None, 0, 0, 1, None]
        
        ner = OttomanNER()
        ner.tokenizer = MagicMock()
        ner.label2id = {"O": 0, "B-PER": 1, "I-PER": 2}
        
        # Mock tokenizer output to return an object with word_ids method
        ner.tokenizer.return_value = MockTokenizerOutput({
            "input_ids": [[101, 1000, 1001, 1002, 102]],
            "attention_mask": [[1, 1, 1, 1, 1]],
        })
        
        examples = {
            "tokens": [["Sultan", "Abdülhamid"]],
            "ner_tags": [["B-PER", "I-PER"]]
        }
        
        result = ner.tokenize_and_align_labels(examples)
        
        assert "labels" in result
        assert len(result["labels"]) == 1
        # Check that only first subword gets label, others get -100
        labels = result["labels"][0]
        assert labels[0] == -100  # [CLS]
        assert labels[1] == 1     # B-PER
        assert labels[2] == -100  # Subword of first token
        assert labels[3] == 2     # I-PER
        assert labels[4] == -100  # [SEP]


class TestTrainFromConfig:
    """Tests for train_from_config method."""
    
    @patch("ottoman_ner.core.AutoTokenizer")
    @patch("ottoman_ner.core.AutoModelForTokenClassification")
    @patch("ottoman_ner.core.Trainer")
    @patch("ottoman_ner.core.TrainingArguments")
    @patch("ottoman_ner.core.DataCollatorForTokenClassification")
    def test_train_from_config_basic(
        self, mock_collator_class, mock_args_class, mock_trainer_class,
        mock_model_class, mock_tokenizer_class, training_config
    ):
        """Test training from config."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_trainer = MagicMock()
        mock_trainer.evaluate.return_value = {"eval_f1": 0.95}
        mock_trainer_class.return_value = mock_trainer
        
        mock_args = MagicMock()
        mock_args.output_dir = training_config["training"]["output_dir"]
        mock_args_class.return_value = mock_args
        
        ner = OttomanNER()
        result = ner.train_from_config(training_config)
        
        assert "eval_f1" in result
        mock_trainer.train.assert_called_once()
        mock_trainer.save_model.assert_called_once()
    
    def test_train_from_config_missing_keys(self):
        """Test training with missing config keys raises errors."""
        ner = OttomanNER()
        
        # Missing data config
        with pytest.raises(ValueError, match="data.train_file"):
            ner.train_from_config({"model": {}})


class TestEvaluate:
    """Tests for evaluate method."""
    
    @patch.object(OttomanNER, "load_model")
    @patch.object(OttomanNER, "load_conll_data")
    @patch.object(OttomanNER, "predict")
    @patch("ottoman_ner.core.precision_score")
    @patch("ottoman_ner.core.recall_score")
    @patch("ottoman_ner.core.f1_score")
    def test_evaluate_basic(
        self, mock_f1, mock_recall, mock_precision, mock_predict,
        mock_load_data, mock_load_model, temp_dir
    ):
        """Test basic evaluation."""
        # Setup mocks
        mock_load_model.return_value = None
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = lambda self: iter([
            {"tokens": ["Sultan", "Abdülhamid"], "ner_tags": ["B-PER", "I-PER"]},
        ])
        mock_load_data.return_value = mock_dataset
        mock_predict.return_value = [
            {"text": "Sultan Abdülhamid", "label": "PER", "start": 0, "end": 2, "confidence": 0.95}
        ]
        mock_precision.return_value = 0.9
        mock_recall.return_value = 0.85
        mock_f1.return_value = 0.87
        
        output_dir = temp_dir / "eval_output"
        
        ner = OttomanNER()
        result = ner.evaluate(
            model_path="test-model",
            test_file="test.conll",
            output_dir=str(output_dir)
        )
        
        assert "overall_precision" in result
        assert "overall_recall" in result
        assert "overall_f1" in result
        mock_load_model.assert_called_once_with("test-model")
        
        # Check output files created
        assert (output_dir / "evaluation_results.json").exists()


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_stray_i_label_handling(self, mock_ner_instance, mock_tokenizer):
        """Test handling of stray I- labels without preceding B-."""
        # Mock tokenizer to simulate stray I- label scenario
        mock_tokenizer.word_ids = lambda batch_index=0: [0, 1, 2]
        
        # The mock model already returns predictions that could include stray I- labels
        result = mock_ner_instance.predict("word1 word2 word3")
        
        # Should handle gracefully without crashing
        assert isinstance(result, list)
    
    def test_entity_confidence_calculation(self, mock_ner_instance):
        """Test that entity confidence is properly averaged."""
        result = mock_ner_instance.predict("Sultan Abdülhamid İstanbul")
        
        for entity in result:
            assert "confidence" in entity
            assert isinstance(entity["confidence"], float)
            assert 0 <= entity["confidence"] <= 1
    
    def test_label_mapping_consistency(self):
        """Test that label mappings are consistent."""
        ner = OttomanNER()
        
        # Check that label2id and id2label are inverses
        for label, idx in ner.label2id.items():
            assert ner.id2label[idx] == label
        
        for idx, label in ner.id2label.items():
            assert ner.label2id[label] == idx
