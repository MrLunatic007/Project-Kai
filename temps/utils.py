import numpy as np

def sigmoid(x):
    """Sigmoid activation function with clipping for numerical stability"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    return x * (1 - x)

def softmax(x):
    """Softmax function with numerical stability"""
    # Shift for numerical stability
    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU function"""
    return np.where(x > 0, 1, 0)

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return np.dot(v1, v2) / (norm1 * norm2)

def levenshtein_distance(s1, s2):
    """Compute the Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Calculate insertions, deletions and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]