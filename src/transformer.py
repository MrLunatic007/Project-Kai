import numpy as np
from src.utils import softmax

class LightTransformer:
    """A lightweight transformer model suitable for running on resource-constrained devices"""
    
    def __init__(self, input_size, hidden_size, num_heads, num_layers, output_size):
        self.input_size = input_size
        self.hidden_size = input_size  # Match input_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.query_weights = [np.random.randn(input_size, input_size) * np.sqrt(2.0 / input_size) for _ in range(num_layers)]
        self.key_weights = [np.random.randn(input_size, input_size) * np.sqrt(2.0 / input_size) for _ in range(num_layers)]
        self.value_weights = [np.random.randn(input_size, input_size) * np.sqrt(2.0 / input_size) for _ in range(num_layers)]
        self.ffn_weights1 = [np.random.randn(input_size, input_size * 2) * np.sqrt(2.0 / input_size) for _ in range(num_layers)]
        self.ffn_weights2 = [np.random.randn(input_size * 2, input_size) * np.sqrt(2.0 / (input_size * 2)) for _ in range(num_layers)]
        self.output_weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
    
    def attention(self, query, key, value):
        attention_scores = np.dot(query, key.T) / np.sqrt(self.input_size)
        attention_weights = softmax(attention_scores)
        return np.dot(attention_weights, value)
    
    def forward(self, X):
        batch_size = X.shape[0]
        hidden = X
        
        for layer in range(self.num_layers):
            q = np.dot(hidden, self.query_weights[layer])
            k = np.dot(hidden, self.key_weights[layer])
            v = np.dot(hidden, self.value_weights[layer])
            attention_output = np.zeros((batch_size, self.input_size))
            head_size = self.input_size // self.num_heads
            for h in range(self.num_heads):
                start = h * head_size
                end = (h + 1) * head_size if h < self.num_heads - 1 else self.input_size
                q_head = q[:, start:end]
                k_head = k[:, start:end]
                v_head = v[:, start:end]
                attention_output[:, start:end] = self.attention(q_head, k_head, v_head)
            hidden = hidden + attention_output
            ffn_output = np.dot(hidden, self.ffn_weights1[layer])
            ffn_output = np.maximum(0, ffn_output)
            ffn_output = np.dot(ffn_output, self.ffn_weights2[layer])
            hidden = hidden + ffn_output
        
        logits = np.dot(hidden, self.output_weights)
        return logits
    
    def train(self, X, y, learning_rate=0.01, epochs=10, batch_size=32):
        num_samples = X.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            total_loss = 0
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                logits = self.forward(X_batch)
                probs = softmax(logits)
                y_one_hot = np.zeros((len(y_batch), self.output_size))
                for i, label in enumerate(y_batch):
                    y_one_hot[i, label] = 1
                
                # Corrected loss calculation
                batch_loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-10), axis=1))
                total_loss += batch_loss
                
                dloss = (probs - y_one_hot) / len(y_batch)  # Normalize by batch size
                self._update_weights(X_batch, dloss, learning_rate)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.4f}")
    
    def _update_weights(self, X, dloss, learning_rate):
        batch_size = X.shape[0]
        hidden = X
        
        # Output layer gradient
        d_output = dloss
        d_hidden = np.dot(d_output, self.output_weights.T)
        self.output_weights -= learning_rate * np.dot(hidden.T, d_output)
        
        for layer in range(self.num_layers - 1, -1, -1):
            d_ffn_out = d_hidden
            ffn_out = np.dot(hidden, self.ffn_weights1[layer])
            ffn_out = np.maximum(0, ffn_out)
            d_ffn_in = np.dot(d_ffn_out, self.ffn_weights2[layer].T) * (ffn_out > 0)
            self.ffn_weights2[layer] -= learning_rate * np.dot(ffn_out.T, d_ffn_out)
            self.ffn_weights1[layer] -= learning_rate * np.dot(hidden.T, d_ffn_in)
            d_hidden += d_ffn_out
            
            q = np.dot(hidden, self.query_weights[layer])
            k = np.dot(hidden, self.key_weights[layer])
            v = np.dot(hidden, self.value_weights[layer])
            attention_output = np.zeros((batch_size, self.input_size))
            head_size = self.input_size // self.num_heads
            for h in range(self.num_heads):
                start = h * head_size
                end = (h + 1) * head_size if h < self.num_heads - 1 else self.input_size
                q_head = q[:, start:end]
                k_head = k[:, start:end]
                v_head = v[:, start:end]
                attention_scores = np.dot(q_head, k_head.T) / np.sqrt(self.input_size)
                attention_weights = softmax(attention_scores)
                attention_output[:, start:end] = np.dot(attention_weights, v_head)
            
            d_attention = d_hidden
            hidden = hidden + attention_output
            self.query_weights[layer] -= learning_rate * np.dot(hidden.T, d_attention)
            self.key_weights[layer] -= learning_rate * np.dot(hidden.T, d_attention)
            self.value_weights[layer] -= learning_rate * np.dot(hidden.T, d_attention)