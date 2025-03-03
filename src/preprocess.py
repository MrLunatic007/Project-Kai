import numpy as np
import string
import csv
import re

class Preprocessor:
    def __init__(self, embedding_size=50):
        self.vocab = {}
        self.vocab_size = 0
        self.embedding_size = embedding_size
        self.word_embeddings = {}
        self.stop_words = set([
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
            "in", "on", "at", "to", "for", "with", "by", "about", "like",
            "from", "then", "than", "when", "where", "how", "all", "any",
            "both", "each", "few", "more", "most", "other", "some", "such", 
            "that", "this", "these", "those", "only", "very"
        ])
    
    def load_dataset(self, data_file):
        """Load and parse the dataset with improved handling"""
        X, y = [], []
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                # Try to detect CSV format
                sample = f.read(1024)
                f.seek(0)
                
                if ',' in sample:
                    reader = csv.reader(f)
                    header = next(reader, None)  # Skip header if exists
                    
                    for row in reader:
                        if len(row) >= 3:  # Ensure we have input and output columns
                            # Combine first two columns for input, third for output
                            command = f"{row[0]} {row[1] if len(row) > 1 else ''}"
                            response = row[2] if len(row) > 2 else ""
                            
                            # Clean and validate the data
                            command = self._clean_text(command)
                            response = self._clean_text(response)
                            
                            if command and response:
                                X.append(command)
                                y.append(response)
                else:
                    # Handle non-CSV formats
                    lines = f.readlines()
                    for i in range(0, len(lines) - 1, 2):
                        if i + 1 < len(lines):
                            command = self._clean_text(lines[i])
                            response = self._clean_text(lines[i + 1])
                            
                            if command and response:
                                X.append(command)
                                y.append(response)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Provide minimal default dataset if loading fails
            X = ["hello", "how are you", "tell me a story"]
            y = ["Hi there!", "I'm doing well, thanks for asking!", "Once upon a time..."]
            
        print(f"Loaded {len(X)} sample pairs from dataset")
        return X, y
    
    def _clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove non-printable characters
        text = ''.join(c for c in text if c.isprintable())
        
        return text
    
    def build_vocab(self, texts):
        """Build vocabulary from texts with improved handling"""
        # Count word frequencies
        word_counts = {}
        for text in texts:
            if not text:
                continue
                
            words = self._tokenize(text)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Keep words that appear at least twice
        min_count = 2
        filtered_words = [word for word, count in word_counts.items() 
                         if count >= min_count and len(word) > 1]
        
        # Build vocabulary
        self.vocab = {word: idx for idx, word in enumerate(filtered_words)}
        self.vocab_size = len(self.vocab)
        
        # Generate embeddings
        self.word_embeddings = self._generate_embeddings()
        print(f"Built vocabulary with {self.vocab_size} words")
    
    def _tokenize(self, text):
        """Tokenize text into words with improved handling"""
        if not text:
            return []
            
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Split into words and filter stop words
        words = [word for word in text.split() 
                if word and word not in self.stop_words and len(word) > 1]
        
        return words
    
    def _generate_embeddings(self):
        """Generate word embeddings with normalization"""
        np.random.seed(42)  # For reproducibility
        
        # Create embeddings for each word in vocab
        embeddings = {}
        for word in self.vocab:
            # Generate random vector
            vec = np.random.randn(self.embedding_size)
            
            # Normalize to unit length
            vec = vec / np.sqrt(np.sum(vec**2))
            
            embeddings[word] = vec
            
        return embeddings
    
    def text_to_vector(self, text):
        """Convert text to vector representation more robustly"""
        if not text:
            return np.zeros(self.embedding_size)
            
        # Tokenize text
        words = self._tokenize(text)
        
        if not words:
            return np.zeros(self.embedding_size)
        
        # Get embeddings for known words
        vectors = []
        for word in words:
            if word in self.word_embeddings:
                vectors.append(self.word_embeddings[word])
        
        # If no known words, return zero vector
        if not vectors:
            return np.zeros(self.embedding_size)
        
        # Average the word vectors
        return np.mean(vectors, axis=0)