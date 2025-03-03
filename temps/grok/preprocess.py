import numpy as np
import string
import csv

class Preprocessor:
    def __init__(self, embedding_size=50):
        self.vocab = {}
        self.vocab_size = 0
        self.embedding_size = embedding_size
        self.word_embeddings = {}
        self.stop_words = set(["a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "in", "on", "at", "to", "for", "with", "by"])
    
    def load_dataset(self, data_file):
        """Load and parse the text-only dataset with proper CSV handling"""
        X, y = [], []
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 4:  # Ensure Author,Title,Description,Genre
                    command = f"{row[0]} {row[1]}"  # Author + Title as input
                    response = row[2]  # Full Description as response
                    X.append(command)
                    y.append(response)
        return X, y
    
    def build_vocab(self, texts):
        """Build vocabulary from text-only dataset"""
        all_words = set()
        for text in texts:
            words = self._tokenize(text)
            all_words.update(words)
        self.vocab = {word: idx for idx, word in enumerate(all_words)}
        self.vocab_size = len(self.vocab)
        self.word_embeddings = self._generate_embeddings()
        print(f"Built vocabulary with {self.vocab_size} words")
    
    def _tokenize(self, text):
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        words = [word for word in text.split() if word and word not in self.stop_words]
        return words
    
    def _generate_embeddings(self):
        np.random.seed(42)
        embeddings = {word: np.random.randn(self.embedding_size) / np.sqrt(self.embedding_size) 
                     for word in self.vocab}
        return embeddings
    
    def text_to_vector(self, text):
        words = self._tokenize(text)
        vectors = [self.word_embeddings.get(word, np.zeros(self.embedding_size)) for word in words]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.embedding_size)