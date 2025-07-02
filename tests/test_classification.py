"""
Unit tests for the classification module.
"""

import unittest
from src.classification import BaselineClassifier, ClassificationResult
from src.data_processing import AbstractProcessor


class TestBaselineClassifier(unittest.TestCase):
    """Test cases for BaselineClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = BaselineClassifier()
    
    def test_cancer_abstract_classification(self):
        """Test classification of cancer-related abstract."""
        abstract = "This study investigates the molecular mechanisms of lung cancer progression."
        result = self.classifier.predict_single(abstract, "test123")
        
        self.assertEqual(result.predicted_labels, ["Cancer"])
        self.assertGreater(result.confidence_scores["Cancer"], 0.5)
        self.assertIn("cancer", result.extracted_diseases)
    
    def test_non_cancer_abstract_classification(self):
        """Test classification of non-cancer abstract."""
        abstract = "The effects of exercise on cardiovascular health and blood pressure regulation."
        result = self.classifier.predict_single(abstract, "test456")
        
        self.assertEqual(result.predicted_labels, ["Non-Cancer"])
        self.assertGreater(result.confidence_scores["Non-Cancer"], 0.5)
    
    def test_batch_classification(self):
        """Test batch classification."""
        abstracts = [
            {"pubmed_id": "123", "abstract": "Lung cancer treatment strategies."},
            {"pubmed_id": "456", "abstract": "Diabetes management in elderly patients."}
        ]
        
        results = self.classifier.predict_batch(abstracts)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].predicted_labels, ["Cancer"])
        self.assertEqual(results[1].predicted_labels, ["Non-Cancer"])
    
    def test_format_output(self):
        """Test output formatting."""
        result = ClassificationResult(
            pubmed_id="test123",
            predicted_labels=["Cancer"],
            confidence_scores={"Cancer": 0.85, "Non-Cancer": 0.15},
            extracted_diseases=["lung cancer"],
            raw_probabilities=None
        )
        
        formatted = self.classifier.format_output(result)
        
        self.assertEqual(formatted["pubmed_id"], "test123")
        self.assertEqual(formatted["classification"]["predicted_labels"], ["Cancer"])
        self.assertEqual(formatted["disease_extraction"]["extracted_diseases"], ["lung cancer"])


class TestAbstractProcessor(unittest.TestCase):
    """Test cases for AbstractProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = AbstractProcessor()
    
    def test_clean_abstract(self):
        """Test abstract cleaning."""
        dirty_abstract = "This study [1] investigates cancer. DOI: 10.1234/abc PMID: 12345"
        cleaned = self.processor.clean_abstract(dirty_abstract)
        
        self.assertNotIn("[1]", cleaned)
        self.assertNotIn("DOI:", cleaned)
        self.assertNotIn("PMID:", cleaned)
        self.assertIn("cancer", cleaned)
    
    def test_extract_diseases(self):
        """Test disease extraction."""
        abstract = "Lung cancer and breast cancer are common malignancies."
        diseases = self.processor.extract_diseases(abstract)
        
        self.assertIn("lung cancer", diseases)
        self.assertIn("breast cancer", diseases)
        self.assertIn("cancer", diseases)
    
    def test_is_cancer_related(self):
        """Test cancer detection."""
        cancer_abstract = "This study focuses on melanoma treatment."
        non_cancer_abstract = "This study focuses on diabetes management."
        
        self.assertTrue(self.processor.is_cancer_related(cancer_abstract))
        self.assertFalse(self.processor.is_cancer_related(non_cancer_abstract))


if __name__ == "__main__":
    unittest.main() 