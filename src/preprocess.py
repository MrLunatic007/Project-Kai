import numpy as np
import string
import csv

class Preprocessor:
    def __init__(self, vocab_size=10000, embedding_size=128):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_embeddings = None
        self.stop_words = set(["a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "in", "on", "at", "to", "for", "with", "by"])
    
    def load_dataset(self, data_file):
        """Load and parse CSV dataset"""
        X, y = [], []
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 4:  # Author,Title,Description,Genre
                    command = f"{row[0]} {row[1]}"  # Author + Title
                    response = row[2][:500]  # Cap Description at 500 chars
                    X.append(command)
                    y.append(response)
        return X, y
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        word_counts = {}
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size-2]
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}
        for i, (word, _) in enumerate(sorted_words, 2):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word
        
        self.word_embeddings = np.random.randn(self.vocab_size, self.embedding_size) * 0.01
        print(f"Built vocabulary with {len(self.word_to_idx)} words")
    
    def _tokenize(self, text):
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return [word for word in text.split() if word and word not in self.stop_words]
    
    def text_to_ids(self, text, max_len=50):
        tokens = self._tokenize(text)
        ids = [self.word_to_idx.get(token, 1) for token in tokens[:max_len]]
        return ids + [0] * (max_len - len(ids))
    
    def ids_to_text(self, ids):
        return " ".join(self.idx_to_word.get(id, "<UNK>") for id in ids if id != 0)