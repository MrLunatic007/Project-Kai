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
            "greeting": ["Hey there!", "Hi!", "Hey", "Hello!", "Greetings!", "How can I help?", "What’s up?", "Hey, how can I assist you?", "Hello, what can I do for you?"],
            "farewell": ["Catch you later!", "Bye for now!", "Take it easy!", "Goodbye!", "See you later!", "Farewell!", "Have a good one!"],
            "unknown": ["Hmm, not sure about that.", "I’ll need more info!", "I’m not sure I understand.", "I’m still learning, can you ask something else?", "I’m not quite sure what you mean."]
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
        
        # Store more dataset samples (up to 50)
        self.dataset_samples = y_text[:50]  # Keep more examples for variety
        
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
        
        if "hi" in user_input.lower() or "hello" in user_input.lower():
            response = np.random.choice(self.templates["greeting"])
        elif "bye" in user_input.lower() or "goodbye" in user_input.lower():
            response = np.random.choice(self.templates["farewell"])
        else:
            context_text = " ".join(item["content"] for item in context)
            input_vector = self.preprocessor.text_to_vector(context_text)
            current_state = input_vector
            
            if self.use_reinforcement and self.rl_model and self.last_state is not None:
                reward = 0.1 + (0.3 if len(user_input.split()) > 3 else 0)
                self.rl_model.store_experience(self.last_state, self.last_action, reward, current_state, False)
                
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
            response = self.reasoning_engine.enhance_response(user_input, base_response, context)
        
        context.append({"role": "assistant", "content": response})
        self.current_conversation.append({"role": "assistant", "content": response})
        self.memory_manager.add_memory(user_input, response)
        
        if self.memory_manager.check_retrain_needed():
            self._retrain_from_memory()
        
        return response, context
    
    def _generate_base_response(self, response_type, user_input):
        """Generate a context-aware base response"""
        words = user_input.split()
        topic = words[-1] if words else "stuff"
        sample = np.random.choice(self.dataset_samples)
        
        if response_type == "tell_story":
            return f"Once there was a tale about {topic}: {sample[:100]}..."
        elif response_type == "ask_question":
            return f"What do you think about {topic}?"
        else:  # give_info
            return f"I’ve got info on {topic}: {sample[:100]}..."  # Longer snippet
    
    def _retrain_from_memory(self):
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
        self.dataset_samples = model_data.get('dataset_samples', [])
        self.rl_model = model_data.get('rl_model')
        self.use_reinforcement = self.rl_model is not None