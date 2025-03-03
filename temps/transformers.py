import numpy as np
from src.utils import softmax

class LightTransformer:
    """A lightweight transformer model suitable for running on resource-constrained devices"""
    
    def __init__(self, input_size, hidden_size, num_heads, num_layers, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Initialize weights with He initialization
        self.query_weights = []
        self.key_weights = []
        self.value_weights = []
        self.ffn_weights1 = []
        self.ffn_weights2 = []
        self.output_weights = None
        
        for _ in range(num_layers):
            # Multi-head attention weights
            self.query_weights.append(
                np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            )
            self.key_weights.append(
                np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            )
            self.value_weights.append(
                np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            )
            
            # Feed-forward network weights
            self.ffn_weights1.append(
                np.random.randn(hidden_size, hidden_size * 2) * np.sqrt(2.0 / hidden_size)
            )
            self.ffn_weights2.append(
                np.random.randn(hidden_size * 2, hidden_size) * np.sqrt(2.0 / (hidden_size * 2))
            )
        
        # Output layer
        self.output_weights = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
    
    def attention(self, query, key, value):
        """Compute scaled dot-product attention"""
        # Compute attention scores
        attention_scores = np.dot(query, key.T) / np.sqrt(self.hidden_size)
        
        # Apply softmax to get attention weights
        attention_weights = softmax(attention_scores)
        
        # Apply attention weights to values
        output = np.dot(attention_weights, value)
        
        return output
    
    def forward(self, X):
        """Forward pass through the transformer model"""
        batch_size = X.shape[0]
        hidden = X
        
        for layer in range(self.num_layers):
            # Multi-head attention
            q = np.dot(hidden, self.query_weights[layer])
            k = np.dot(hidden, self.key_weights[layer])
            v = np.dot(hidden, self.value_weights[layer])
            
            # Split heads for multi-head attention
            q_heads = q.reshape(batch_size, self.num_heads, -1)
            k_heads = k.reshape(batch_size, self.num_heads, -1)
            v_heads = v.reshape(batch_size, self.num_heads, -1)
            
            # Compute attention for each head
            attention_output = np.zeros_like(q_heads)
            for b in range(batch_size):
                for h in range(self.num_heads):
                    attention_output[b, h] = self.attention(
                        q_heads[b, h].reshape(1, -1),
                        k_heads[b, h].reshape(1, -1),
                        v_heads[b, h].reshape(1, -1)
                    )
            
            # Combine heads
            attention_output = attention_output.reshape(batch_size, -1)
            
            # Residual connection
            attention_output = hidden + attention_output
            
            # Feed-forward network
            ffn_output = np.dot(attention_output, self.ffn_weights1[layer])
            ffn_output = np.maximum(0, ffn_output)  # ReLU
            ffn_output = np.dot(ffn_output, self.ffn_weights2[layer])
            
            # Residual connection
            hidden = attention_output + ffn_output
        
        # Output layer
        logits = np.dot(hidden, self.output_weights)
        
        return logits
    
    def train(self, X, y, learning_rate=0.01, epochs=10, batch_size=32):
        """Train the transformer model"""
        num_samples = X.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            total_loss = 0
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, num_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                logits = self.forward(X_batch)
                
                # Convert to one-hot encoding
                y_one_hot = np.zeros((len(y_batch), self.output_size))
                for i, label in enumerate(y_batch):
                    y_one_hot[i, label] = 1
                
                # Compute softmax and cross-entropy loss
                probs = softmax(logits)
                batch_loss = -np.sum(y_one_hot * np.log(probs + 1e-10)) / len(y_batch)
                total_loss += batch_loss
                
                # Backward pass (simplified for implementation)
                # Gradient of cross-entropy loss with respect to logits
                dloss = probs - y_one_hot
                
                # Update weights (simplified SGD)
                # In a real implementation, we would compute proper gradients
                # This is a simplified approach for lightweight devices
                self._update_weights(X_batch, dloss, learning_rate)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.4f}")
    
    def _update_weights(self, X, dloss, learning_rate):
        """Update model weights using gradients"""
        # This is a simplified weight update
        # A real implementation would compute proper gradients through the network
        
        # Update output layer weights
        batch_size = X.shape[0]
        hidden = X  # Simplified: use input as hidden state
        
        for layer in range(self.num_layers):
            # Move through the network, updating weights
            # Multi-head attention weights
            grad_factor = 0.01  # Scale gradient for stability
            
            # Update various weights with a simple approximation
            self.query_weights[layer] -= learning_rate * grad_factor * np.dot(hidden.T, dloss)
            self.key_weights[layer] -= learning_rate * grad_factor * np.dot(hidden.T, dloss)
            self.value_weights[layer] -= learning_rate * grad_factor * np.dot(hidden.T, dloss)
            
            # Simplified feed-forward updates
            self.ffn_weights2[layer] -= learning_rate * grad_factor * np.dot(hidden.T, dloss)
            self.ffn_weights1[layer] -= learning_rate * grad_factor * np.dot(hidden.T, dloss)
            
            # Apply non-linearity for next layer
            hidden = np.maximum(0, hidden)  # Simple non-linearity
        
        # Output layer gradient
        self.output_weights -= learning_rate * np.dot(hidden.T, dloss)