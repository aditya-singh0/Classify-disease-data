"""
Data processing utilities for research paper abstracts.
Handles PubMed data cleaning, normalization, and preparation for model training.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split
from datasets import Dataset
import os
import glob
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AbstractProcessor:
    """Process and clean research paper abstracts."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.cancer_keywords = {
            'cancer', 'carcinoma', 'tumor', 'tumour', 'neoplasm', 'malignancy',
            'metastasis', 'metastatic', 'oncology', 'oncological', 'leukemia',
            'lymphoma', 'sarcoma', 'melanoma', 'adenocarcinoma', 'squamous',
            'breast cancer', 'lung cancer', 'prostate cancer', 'colorectal cancer',
            'pancreatic cancer', 'ovarian cancer', 'cervical cancer', 'brain cancer',
            'liver cancer', 'kidney cancer', 'bladder cancer', 'thyroid cancer',
            'gastric cancer', 'esophageal cancer', 'skin cancer', 'bone cancer',
            'blood cancer', 'multiple myeloma', 'hodgkin lymphoma', 'non-hodgkin lymphoma'
        }
        
        # Enhanced disease patterns for better extraction
        self.disease_patterns = {
            'cancer_types': [
                r'\b(?:lung|breast|prostate|colorectal|pancreatic|ovarian|cervical|brain|liver|kidney|bladder|thyroid|gastric|esophageal|skin|bone|blood)\s+cancer\b',
                r'\b(?:cancer|carcinoma|tumor|tumour|neoplasm|malignancy)\b',
                r'\b(?:leukemia|lymphoma|sarcoma|melanoma|adenocarcinoma|squamous)\b',
                r'\b(?:multiple myeloma|hodgkin lymphoma|non-hodgkin lymphoma)\b'
            ],
            'other_diseases': [
                r'\b(?:diabetes|hypertension|asthma|arthritis|alzheimer|parkinson)\b',
                r'\b(?:heart disease|cardiovascular disease|stroke)\b',
                r'\b(?:hiv|aids|tuberculosis|malaria|hepatitis)\b',
                r'\b(?:obesity|diabetes mellitus|type 1 diabetes|type 2 diabetes)\b'
            ]
        }
    
    def clean_abstract(self, text: str) -> str:
        """Clean and normalize abstract text."""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove citations in brackets
        text = re.sub(r'\[[^\]]*\]', '', text)
        
        # Remove DOI references
        text = re.sub(r'doi:\s*[^\s]+', '', text)
        
        # Remove PubMed IDs
        text = re.sub(r'pmid:\s*\d+', '', text)
        
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_diseases(self, text: str) -> List[str]:
        """Extract disease mentions from abstract text."""
        diseases = []
        text_lower = text.lower()
        
        # Extract all disease types
        for category, patterns in self.disease_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                diseases.extend(matches)
        
        # Remove duplicates and sort
        diseases = list(set(diseases))
        diseases.sort()
        
        return diseases
    
    def is_cancer_related(self, text: str) -> bool:
        """Determine if abstract is cancer-related based on keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.cancer_keywords)
    
    def tokenize_text(self, text: str, max_length: int = 512) -> List[str]:
        """Tokenize text and limit to max_length."""
        tokens = word_tokenize(text)
        return tokens[:max_length]


class DatasetProcessor:
    """Process and prepare datasets for model training."""
    
    def __init__(self, abstract_processor: AbstractProcessor):
        self.abstract_processor = abstract_processor
    
    def load_pubmed_data(self, file_path: str) -> pd.DataFrame:
        """Load PubMed data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} abstracts from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def load_dataset_from_files(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset from Cancer and Non-Cancer folders."""
        logger.info(f"Loading dataset from {dataset_path}")
        
        data = []
        
        # Load Cancer files
        cancer_path = os.path.join(dataset_path, "Cancer")
        if os.path.exists(cancer_path):
            cancer_files = glob.glob(os.path.join(cancer_path, "*.txt"))
            logger.info(f"Found {len(cancer_files)} cancer files")
            
            for file_path in cancer_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    # Parse the file content
                    pubmed_id, title, abstract = self._parse_pubmed_file(content)
                    
                    data.append({
                        'pubmed_id': pubmed_id,
                        'title': title,
                        'abstract': abstract,
                        'label': 1,  # Cancer
                        'label_text': 'Cancer'
                    })
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
        
        # Load Non-Cancer files
        non_cancer_path = os.path.join(dataset_path, "Non-Cancer")
        if os.path.exists(non_cancer_path):
            non_cancer_files = glob.glob(os.path.join(non_cancer_path, "*.txt"))
            logger.info(f"Found {len(non_cancer_files)} non-cancer files")
            
            for file_path in non_cancer_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    # Parse the file content
                    pubmed_id, title, abstract = self._parse_pubmed_file(content)
                    
                    data.append({
                        'pubmed_id': pubmed_id,
                        'title': title,
                        'abstract': abstract,
                        'label': 0,  # Non-Cancer
                        'label_text': 'Non-Cancer'
                    })
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
        
        df = pd.DataFrame(data)
        logger.info(f"Total loaded: {len(df)} abstracts")
        return df
    
    def _parse_pubmed_file(self, content: str) -> Tuple[str, str, str]:
        """Parse PubMed file content to extract ID, title, and abstract."""
        lines = content.split('\n')
        
        pubmed_id = "unknown"
        title = ""
        abstract = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('<ID:'):
                pubmed_id = line.replace('<ID:', '').replace('>', '').strip()
            elif line.startswith('Title:'):
                title = line.replace('Title:', '').strip()
            elif line.startswith('Abstract:'):
                abstract = line.replace('Abstract:', '').strip()
        
        return pubmed_id, title, abstract
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the dataset."""
        logger.info("Cleaning dataset...")
        
        # Remove rows with missing abstracts
        df = df.dropna(subset=['abstract'])
        
        # Clean abstracts
        df['cleaned_abstract'] = df['abstract'].apply(self.abstract_processor.clean_abstract)
        
        # Remove empty abstracts after cleaning
        df = df[df['cleaned_abstract'].astype(str).str.len() > 10]
        
        # Extract diseases
        df['extracted_diseases'] = df['cleaned_abstract'].apply(
            self.abstract_processor.extract_diseases
        )
        
        # Create binary labels if not present
        if 'label' not in df.columns:
            df['label'] = df['cleaned_abstract'].apply(
                lambda x: 1 if self.abstract_processor.is_cancer_related(x) else 0
            )
        
        # Map labels to text
        df['label_text'] = df['label'].astype(int).map({1: 'Cancer', 0: 'Non-Cancer'})
        
        logger.info(f"Cleaned dataset: {len(df)} abstracts remaining")
        logger.info(f"Cancer samples: {len(df[df['label'] == 1])}")
        logger.info(f"Non-Cancer samples: {len(df[df['label'] == 0])}")
        
        return df
    
    def prepare_for_training(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
        """Prepare dataset for model training with train/validation/test splits."""
        logger.info("Preparing dataset for training...")
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['label']
        )
        
        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size/(1-test_size), random_state=42, stratify=train_val_df['label']
        )
        
        # Create HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        logger.info(f"Training set: {len(train_dataset)} samples")
        logger.info(f"Validation set: {len(val_dataset)} samples")
        logger.info(f"Test set: {len(test_dataset)} samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_sample_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """Create sample data for testing and demonstration."""
        logger.info(f"Creating {num_samples} sample abstracts...")
        
        cancer_abstracts = [
            "This study investigates the molecular mechanisms of lung cancer progression and identifies novel therapeutic targets.",
            "Breast cancer screening methods and early detection strategies in high-risk populations.",
            "The role of immunotherapy in treating metastatic melanoma and improving patient outcomes.",
            "Genetic mutations associated with colorectal cancer development and progression.",
            "Novel biomarkers for prostate cancer diagnosis and prognosis prediction.",
            "Targeted therapy approaches for pancreatic cancer treatment and patient survival.",
            "Immunotherapy response prediction in non-small cell lung cancer patients.",
            "Epigenetic modifications in ovarian cancer and their therapeutic implications."
        ]
        
        non_cancer_abstracts = [
            "The effects of exercise on cardiovascular health and blood pressure regulation.",
            "Diabetes management strategies and glycemic control in elderly patients.",
            "Antibiotic resistance patterns in bacterial infections and treatment implications.",
            "The impact of diet on gut microbiome composition and metabolic health.",
            "Sleep quality and its relationship to cognitive function and memory retention.",
            "Hypertension treatment protocols and blood pressure monitoring strategies.",
            "Asthma management in pediatric populations and treatment outcomes.",
            "Arthritis treatment options and patient quality of life improvements."
        ]
        
        data = []
        for i in range(num_samples):
            if i < num_samples // 2:
                # Cancer abstracts
                abstract = np.random.choice(cancer_abstracts)
                label = 1
                label_text = "Cancer"
            else:
                # Non-cancer abstracts
                abstract = np.random.choice(non_cancer_abstracts)
                label = 0
                label_text = "Non-Cancer"
            
            pubmed_id = f"PMID{i+1:06d}"
            
            data.append({
                'pubmed_id': pubmed_id,
                'title': f"Sample Research Paper {i+1}",
                'abstract': abstract,
                'label': label,
                'label_text': label_text
            })
        
        df = pd.DataFrame(data)
        
        # Clean the sample data
        df = self.clean_dataset(df)
        
        logger.info(f"Created sample dataset with {len(df)} abstracts")
        return df


def create_sample_dataset():
    """Create and save a sample dataset for testing."""
    processor = AbstractProcessor()
    dataset_processor = DatasetProcessor(processor)
    
    # Create sample data
    df = dataset_processor.create_sample_data(1000)
    
    # Save to CSV
    output_path = "data/sample_abstracts.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Sample dataset saved to {output_path}")
    return df


if __name__ == "__main__":
    create_sample_dataset() 