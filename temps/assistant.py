import numpy as np
import pickle
import os
from src.transformer import LightTransformer
from src.preprocess import Preprocessor
from src.reasoning import ReasoningEngine

class PersonalAssistant:
    def __init__(self, model_path="models/kai_model.pkl"):
        self.preprocessor = Preprocessor()
        self.model_path = model_path
        self.transformer = None
        self.reasoning_engine = ReasoningEngine()
        self.templates = {
            "greeting": ["Hello!", "Hi there!", "Greetings!"],
            "farewell": ["Goodbye!", "See you later!", "Take care!"],
            "unknown": ["I'm not sure about that.", "I don't have enough information to answer that."]
        }
        self.max_context_length = 5  # Limit context to last 5 turns
    
    def train(self, data_file):
        """Train the model on the provided dataset"""
        # Build vocabulary from dataset
        X_text, y_text = self.preprocessor.load_dataset(data_file)
        self.preprocessor.build_vocab(X_text)
        
        # Convert text to embeddings/vectors
        X = [self.preprocessor.text_to_vector(x) for x in X_text]
        X = np.array(X)
        
        # Create response mappings
        self.response_map = {i: response for i, response in enumerate(set(y_text))}
        y = [list(self.response_map.keys()).index(y_text[i]) for i in range(len(y_text))]
        y = np.array(y)
        
        # Initialize and train transformer
        input_size = self.preprocessor.embedding_size
        self.transformer = LightTransformer(
            input_size=input_size, 
            hidden_size=64,
            num_heads=4,
            num_layers=2,
            output_size=len(self.response_map)
        )
        self.transformer.train(X, y, epochs=50, batch_size=16)
        
        # Save the model
        self.save_model()
    
    def save_model(self):
        """Save model and preprocessor to disk"""
        model_data = {
            'preprocessor': self.preprocessor,
            'transformer': self.transformer,
            'response_map': self.response_map
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self):
        """Load model and preprocessor from disk"""
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.preprocessor = model_data['preprocessor']
        self.transformer = model_data['transformer']
        self.response_map = model_data['response_map']
    
    def process_input(self, user_input, context=None):
        """Process user input, generate response based on context"""
        if context is None:
            context = []
        
        # Update context with user input
        context.append({"role": "user", "content": user_input})
        
        # Keep context to a manageable size
        if len(context) > self.max_context_length * 2:  # * 2 because we have user and assistant turns
            context = context[-self.max_context_length * 2:]
        
        # Check for predefined responses
        if any(greeting in user_input.lower() for greeting in ["hello", "hi", "hey"]):
            response = np.random.choice(self.templates["greeting"])
        elif any(farewell in user_input.lower() for farewell in ["bye", "goodbye", "see you"]):
            response = np.random.choice(self.templates["farewell"])
        else:
            # Prepare input for model using context
            context_text = " ".join([item["content"] for item in context])
            input_vector = self.preprocessor.text_to_vector(context_text)
            
            # Get prediction from model
            if self.transformer:
                logits = self.transformer.forward(np.array([input_vector]))
                pred_idx = np.argmax(logits[0])
                
                # Apply reasoning to refine the response
                base_response = self.response_map.get(pred_idx, np.random.choice(self.templates["unknown"]))
                response = self.reasoning_engine.enhance_response(user_input, base_response, context)
            else:
                response = "I'm not trained yet. Please train me with a dataset first."
        
        # Update context with assistant response
        context.append({"role": "assistant", "content": response})
        
        return response, context