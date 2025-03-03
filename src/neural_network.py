import numpy as np
from src.utils import sigmoid, sigmoid_derivative, softmax

class RLNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, memory_capacity=1000):
        # Network architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with small random values
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
        
        # Reinforcement learning parameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.3  # Exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.learning_rate = 0.01
        
        # Experience memory
        self.memory_capacity = memory_capacity
        self.memory = []
        self.memory_counter = 0
        self.batch_size = 32  # Size of minibatch for training
        
        # Training threshold
        self.train_threshold = 100  # Minimum experiences before training
        self.retrain_frequency = 50  # How often to retrain (in steps)
        self.step_counter = 0
    
    def forward(self, X):
        """Forward pass through the network"""
        # Ensure X is a 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy"""
        # Explore: random action
        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, self.output_size)
        
        # Exploit: best action according to model
        actions_value = self.forward(state)
        return np.argmax(actions_value)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store an experience in memory"""
        experience = (state, action, reward, next_state, done)
        
        if len(self.memory) < self.memory_capacity:
            self.memory.append(experience)
        else:
            # Replace old memory with new one
            index = self.memory_counter % self.memory_capacity
            self.memory[index] = experience
        
        self.memory_counter += 1
        self.step_counter += 1
        
        # Check if we should retrain the model
        if (len(self.memory) >= self.train_threshold and 
            self.step_counter % self.retrain_frequency == 0):
            self.replay()
            
            # Decay exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    def replay(self, batch_size=None):
        """Train the model on a batch of experiences"""
        if batch_size is None:
            batch_size = min(self.batch_size, len(self.memory))
        
        # Sample a minibatch
        mini_batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = np.zeros((batch_size, self.input_size))
        next_states = np.zeros((batch_size, self.input_size))
        actions, rewards, dones = [], [], []
        
        # Extract data from experiences
        for i, idx in enumerate(mini_batch):
            states[i] = self.memory[idx][0]
            actions.append(self.memory[idx][1])
            rewards.append(self.memory[idx][2])
            next_states[i] = self.memory[idx][3]
            dones.append(self.memory[idx][4])
        
        # Current Q values
        current_q = self.forward(states)
        
        # Next Q values
        next_q = self.forward(next_states)
        max_next_q = np.max(next_q, axis=1)
        
        # Update Q values for the actions taken
        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * max_next_q[i]
            
            # Update only the Q value for the action taken
            current_q[i, actions[i]] = target
        
        # Train the network
        self._train_on_batch(states, current_q)
        
        print(f"Replayed {batch_size} experiences, memory size: {len(self.memory)}")
        
        # If memory is full, consider trimming old data
        if len(self.memory) >= self.memory_capacity:
            self._prune_memory()
    
    def _train_on_batch(self, X, y):
        """Train the network on a batch of data"""
        # Forward pass
        self.forward(X)
        
        # Backward pass
        error = y - self.a2
        delta2 = error * sigmoid_derivative(self.a2)
        error_hidden = np.dot(delta2, self.weights2.T)
        delta1 = error_hidden * sigmoid_derivative(self.a1)
        
        # Update weights
        self.weights2 += self.learning_rate * np.dot(self.a1.T, delta2)
        self.bias2 += self.learning_rate * np.sum(delta2, axis=0, keepdims=True)
        self.weights1 += self.learning_rate * np.dot(X.T, delta1)
        self.bias1 += self.learning_rate * np.sum(delta1, axis=0, keepdims=True)
    
    def _prune_memory(self):
        """Remove older experiences to make room for new ones"""
        # Keep the most recent 75% of experiences
        memory_size = len(self.memory)
        keep_size = int(memory_size * 0.75)
        self.memory = self.memory[memory_size - keep_size:]
        print(f"Pruned memory from {memory_size} to {keep_size} experiences")
    
    def save_model(self, filename):
        """Save model weights and parameters"""
        model_data = {
            'weights1': self.weights1,
            'weights2': self.weights2,
            'bias1': self.bias1,
            'bias2': self.bias2,
            'epsilon': self.epsilon
        }
        np.save(filename, model_data)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load model weights and parameters"""
        model_data = np.load(filename, allow_pickle=True).item()
        self.weights1 = model_data['weights1']
        self.weights2 = model_data['weights2']
        self.bias1 = model_data['bias1']
        self.bias2 = model_data['bias2']
        self.epsilon = model_data['epsilon']
        print(f"Model loaded from {filename}")