import numpy as np
from src.utils import softmax, sigmoid_derivative

class ReasoningEngine:
    def __init__(self, vocab_size, embedding_size=128, hidden_size=256):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.intent_weights = np.random.randn(embedding_size, hidden_size) * 0.01
        self.intent_bias = np.zeros((1, hidden_size))
        self.output_weights = np.random.randn(hidden_size, 3) * 0.01  # 3 intents
        self.output_bias = np.zeros((1, 3))
    
    def forward(self, embeddings):
        hidden = np.tanh(np.dot(embeddings, self.intent_weights) + self.intent_bias)
        logits = np.dot(hidden, self.output_weights) + self.output_bias
        probs = softmax(logits)
        return probs
    
    def train(self, X, y, epochs=10, lr=0.001, batch_size=32):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                probs = self.forward(batch_X)
                loss = -np.mean(np.sum(batch_y * np.log(probs + 1e-10), axis=1))
                total_loss += loss
                
                dloss = (probs - batch_y) / len(batch_X)
                d_hidden = np.dot(dloss, self.output_weights.T)
                d_tanh = d_hidden * (1 - np.tanh(np.dot(batch_X, self.intent_weights) + self.intent_bias)**2)
                
                self.output_weights -= lr * np.dot(np.tanh(np.dot(batch_X, self.intent_weights) + self.intent_bias).T, dloss)
                self.output_bias -= lr * np.sum(dloss, axis=0, keepdims=True)
                self.intent_weights -= lr * np.dot(batch_X.T, d_tanh)
                self.intent_bias -= lr * np.sum(d_tanh, axis=0, keepdims=True)
            
            print(f"Reasoning Epoch {epoch+1}/{epochs}, Loss: {total_loss / (len(X) // batch_size):.4f}")

class Generator:
    def __init__(self, vocab_size, embedding_size=128, hidden_size=256, num_layers=2, max_len=50):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len
        
        self.embeddings = np.random.randn(vocab_size, embedding_size) * 0.01
        self.Wq = [np.random.randn(hidden_size, hidden_size) * 0.01 for _ in range(num_layers)]
        self.Wk = [np.random.randn(hidden_size, hidden_size) * 0.01 for _ in range(num_layers)]
        self.Wv = [np.random.randn(hidden_size, hidden_size) * 0.01 for _ in range(num_layers)]
        self.Wo = np.random.randn(hidden_size, vocab_size) * 0.01
    
    def forward(self, input_ids, hidden=None):
        if hidden is None:
            hidden = [np.zeros((len(input_ids), self.hidden_size)) for _ in range(self.num_layers)]
        
        embeddings = self.embeddings[input_ids]
        for layer in range(self.num_layers):
            Q = np.dot(hidden[layer], self.Wq[layer])
            K = np.dot(embeddings, self.Wk[layer])
            V = np.dot(embeddings, self.Wv[layer])
            
            attention_scores = np.dot(Q, K.T) / np.sqrt(self.hidden_size)
            attention_weights = softmax(attention_scores)
            attention_output = np.dot(attention_weights, V)
            hidden[layer] = hidden[layer] + attention_output
        
        logits = np.dot(hidden[-1], self.Wo)
        return logits, hidden
    
    def generate(self, input_ids, max_new_tokens=50):
        hidden = None
        output_ids = list(input_ids)
        
        for _ in range(max_new_tokens):
            logits, hidden = self.forward(np.array([output_ids[-self.max_len:]]), hidden)
            probs = softmax(logits[-1])
            next_id = np.argmax(probs)
            if next_id == 0:  # <PAD>
                break
            output_ids.append(next_id)
        
        return output_ids
    
    def train(self, X, y, epochs=10, lr=0.001):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                input_ids = X[i]
                target_ids = y[i]
                
                hidden = None
                loss = 0
                for t in range(len(target_ids)):
                    logits, hidden = self.forward(np.array([input_ids[-self.max_len:]]), hidden)
                    probs = softmax(logits[-1])
                    loss += -np.log(probs[target_ids[t]] + 1e-10)
                    input_ids.append(target_ids[t])
                
                total_loss += loss / len(target_ids)
                self._backprop(X[i], target_ids, lr)
            
            print(f"Generator Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(X):.4f}")
    
    def _backprop(self, input_ids, target_ids, lr):
        # Placeholder—full backprop is complex, focusing on forward for now
        pass  # TODO: Implement if generation needs refinement