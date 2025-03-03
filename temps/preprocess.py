import numpy as np
import re
import string

class Preprocessor:
    def __init__(self, embedding_size=100):
        self.vocab = {}
        self.vocab_size = 0
        self.embedding_size = embedding_size
        self.word_embeddings = {}
        self.stop_words = set([
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
            "in", "on", "at", "to", "for", "with", "by", "about", "from"
        ])
    
    def load_dataset(self, data_file):
        """Load and parse the dataset"""
        X, y = [], []
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if "|" in line:
                parts = line.strip().split("|", 1)
                if len(parts) == 2:
                    command, response = parts
                    X.append(command)
                    y.append(response)
        
        return X, y
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        # Clean and tokenize all texts
        all_words = set()
        for text in texts:
            words = self._tokenize(text)
            all_words.update(words)
        
        # Create vocabulary
        self.vocab = {word: idx for idx, word in enumerate(all_words)}
        self.vocab_size = len(self.vocab)
        
        # Generate simple word embeddings
        self.word_embeddings = self._generate_embeddings()
    
    def _tokenize(self, text):
        """Tokenize and clean text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Split into words
        words = text.split()
        
        # Remove stop words
        words = [word for word in words if word not in self.stop_words]
        
        return words
    
    def _generate_embeddings(self):
        """Generate simple word embeddings"""
        np.random.seed(42)  # For reproducibility
        embeddings = {}
        
        for word in self.vocab:
            # Generate a random embedding vector for each word
            embeddings[word] = np.random.randn(self.embedding_size)
            # Normalize to unit length
            embeddings[word] = embeddings[word] / np.linalg.norm(embeddings[word])
        
        return embeddings
    
    def text_to_vector(self, text):
        """Convert text to an embedding vector"""
        words = self._tokenize(text)
        
        # If no words match vocabulary, return zeros
        if not any(word in self.word_embeddings for word in words):
            return np.zeros(self.embedding_size)
        
        # Compute average of word embeddings
        vectors = [self.word_embeddings[word] for word in words if word in self.word_embeddings]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.embedding_size)