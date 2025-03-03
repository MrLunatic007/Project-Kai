import numpy as np
import pickle
import os
from src.transformer import LightTransformer
from src.preprocess import Preprocessor
from src.reasoning import ReasoningEngine
from src.neural_network import RLNeuralNetwork
from src.memory_manager import MemoryManager

class PersonalAssistant:
    def __init__(self, model_path="models/kai_model.pkl", use_reinforcement=True):
        self.preprocessor = Preprocessor()
        self.model_path = model_path
        self.transformer = None
        self.rl_model = None
        self.reasoning_engine = ReasoningEngine()
        self.use_reinforcement = use_reinforcement
        self.memory_manager = MemoryManager(max_memories=500)
        self.templates = {
            "greeting": ["Hey there!", "Hi!", "Hello friend!"],
            "farewell": ["Catch you later!", "Bye for now!", "Take it easy!"],
            "unknown": ["Hmm, not sure about that.", "I'll need more info!"]
        }
        self.response_types = ["tell_story", "ask_question", "give_info"]
        self.max_context_length = 3
        self.last_state = None
        self.last_action = None
        self.current_conversation = []
        self.dataset_samples = []  # Store dataset examples
    
    def train(self, data_file):
        """Train Kai with simplified response types"""
        X_text, y_text = self.preprocessor.load_dataset(data_file)
        self.preprocessor.build_vocab(X_text + y_text)
        
        X = np.array([self.preprocessor.text_to_vector(x) for x in X_text])
        y = np.array([0 if "story" in x.lower() else 1 if "how" in x.lower() or "what" in x.lower() else 2 for x in X_text])
        self.response_map = {0: "tell_story", 1: "ask_question", 2: "give_info"}
        
        # Store coherent dataset samples separately by type
        self.dataset_samples = {
            "tell_story": [sample for i, sample in enumerate(y_text[:100]) if y[i % len(y)] == 0],
            "ask_question": [sample for i, sample in enumerate(y_text[:100]) if y[i % len(y)] == 1],
            "give_info": [sample for i, sample in enumerate(y_text[:100]) if y[i % len(y)] == 2]
        }
        
        # Make sure each category has at least some samples
        for key in self.response_types:
            if not self.dataset_samples.get(key):
                self.dataset_samples[key] = ["I don't have much information about that yet."]
        
        self.transformer = LightTransformer(
            input_size=self.preprocessor.embedding_size, 
            hidden_size=50,
            num_heads=2,
            num_layers=1,
            output_size=len(self.response_types)
        )
        self.transformer.train(X, y, epochs=50, batch_size=8, learning_rate=0.005)
        
        if self.use_reinforcement:
            self.rl_model = RLNeuralNetwork(
                input_size=self.preprocessor.embedding_size,
                hidden_size=16,
                output_size=len(self.response_types),
                memory_capacity=200
            )
        self.save_model()
    
    def process_input(self, user_input, context=None):
        if context is None:
            context = []
        context.append({"role": "user", "content": user_input})
        self.current_conversation.append({"role": "user", "content": user_input})
        if len(context) > self.max_context_length * 2:
            context = context[-self.max_context_length * 2:]
        
        # Handle greetings and farewells directly
        if any(word in user_input.lower() for word in ["hi", "hello", "hey", "greetings"]):
            response = np.random.choice(self.templates["greeting"])
        elif any(word in user_input.lower() for word in ["bye", "goodbye", "see you", "farewell"]):
            response = np.random.choice(self.templates["farewell"])
        else:
            # Process normal conversation
            context_text = " ".join(item["content"] for item in context)
            input_vector = self.preprocessor.text_to_vector(context_text)
            current_state = input_vector
            
            # Choose response type using transformer model or reinforcement learning
            if self.use_reinforcement and self.rl_model and self.last_state is not None:
                # Calculate reward based on user engagement
                reward = 0.1 + (0.3 if len(user_input.split()) > 3 else 0)
                self.rl_model.store_experience(self.last_state, self.last_action, reward, current_state, False)
                
                # Choose action: 70% from transformer, 30% from RL
                if np.random.random() < 0.7:
                    logits = self.transformer.forward(np.array([input_vector]))
                    pred_idx = np.argmax(logits[0])
                else:
                    pred_idx = self.rl_model.choose_action(current_state)
                self.last_state = current_state
                self.last_action = pred_idx
            else:
                logits = self.transformer.forward(np.array([input_vector]))
                pred_idx = np.argmax(logits[0])
                self.last_state = current_state
                self.last_action = pred_idx
            
            response_type = self.response_map.get(pred_idx, "give_info")
            base_response = self._generate_base_response(response_type, user_input)
            # Apply reasoning but with sanity check on output length
            raw_response = self.reasoning_engine.enhance_response(user_input, base_response, context)
            
            # Clean up response to ensure coherence
            response = self._cleanup_response(raw_response, response_type)
        
        context.append({"role": "assistant", "content": response})
        self.current_conversation.append({"role": "assistant", "content": response})
        self.memory_manager.add_memory(user_input, response)
        
        if self.memory_manager.check_retrain_needed():
            self._retrain_from_memory()
        
        return response, context
    
    def _generate_base_response(self, response_type, user_input):
        """Generate a coherent base response"""
        # Extract topic from user input more reliably
        words = user_input.lower().split()
        stop_words = {"the", "a", "an", "is", "are", "to", "from", "with", "in", "on", "at", "by", "for"}
        content_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Try to find a topic, defaulting to a generic one if necessary
        topic = content_words[-1] if content_words else "that"
        
        # Get appropriate samples for this response type
        samples = self.dataset_samples.get(response_type, ["I'm learning about that."])
        
        if not samples:
            samples = ["I'm still learning about that topic."]
        
        # Select a random but coherent sample
        sample = np.random.choice(samples)
        
        # Generate specific response based on type
        if response_type == "tell_story":
            # Extract a coherent sentence from the sample
            first_sentence = sample.split('. ')[0] if '. ' in sample else sample
            return f"Here's a short story about {topic}: {first_sentence}."
            
        elif response_type == "ask_question":
            question_templates = [
                f"What do you think about {topic}?",
                f"How do you feel about {topic}?",
                f"What's your experience with {topic}?",
                f"Have you learned much about {topic}?"
            ]
            return np.random.choice(question_templates)
            
        else:  # give_info
            # Extract a coherent piece of information
            info = sample.split('. ')[0] if '. ' in sample else sample
            info = info[:100] + ('...' if len(info) > 100 else '')  # Limit length
            return f"Here's what I know about {topic}: {info}"
    
    def _cleanup_response(self, response, response_type):
        """Clean up the response to ensure coherence"""
        # Remove trailing ellipsis with random text
        if "..." in response:
            parts = response.split("...")
            response = parts[0] + "."
        
        # Cap response length
        if len(response) > 200:
            sentences = response.split('. ')
            response = '. '.join(sentences[:2]) + '.'
        
        # Ensure question mark for questions
        if response_type == "ask_question" and not response.endswith("?"):
            response = response.rstrip(".") + "?"
            
        return response

    def _retrain_from_memory(self):
        """Retrain model from memories of good interactions"""
        training_data = self.memory_manager.get_training_data()
        if not training_data:
            return
        
        X_text = [item["input"] for item in training_data]
        y_text = [item["output"] for item in training_data]
        self.preprocessor.build_vocab(X_text + y_text)
        
        X = np.array([self.preprocessor.text_to_vector(x) for x in X_text])
        y = np.array([0 if "story" in x.lower() else 1 if "how" in x.lower() or "what" in x.lower() else 2 for x in X_text])
        self.transformer.train(X, y, epochs=10, batch_size=4, learning_rate=0.002)
        self.save_model()
        self.memory_manager.clear_short_term_memory()

    def save_model(self):
        model_data = {
            'preprocessor': self.preprocessor,
            'transformer': self.transformer,
            'response_map': self.response_map,
            'rl_model': self.rl_model if self.use_reinforcement else None,
            'dataset_samples': self.dataset_samples
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self):
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.preprocessor = model_data['preprocessor']
        self.transformer = model_data['transformer']
        self.response_map = model_data['response_map']
        self.dataset_samples = model_data.get('dataset_samples', {})
        self.rl_model = model_data.get('rl_model')
        self.use_reinforcement = self.rl_model is not None