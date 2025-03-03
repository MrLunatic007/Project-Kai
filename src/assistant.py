import numpy as np
import pickle
import os
import requests
from bs4 import BeautifulSoup
from src.preprocess import Preprocessor
from src.neural_network import ReasoningEngine, Generator
from src.memory_manager import MemoryManager

class PersonalAssistant:
    def __init__(self, model_path="models/kai_scratch.pkl"):
        self.preprocessor = Preprocessor()
        self.model_path = model_path
        self.generator = None
        self.reasoner = None
        self.memory_manager = MemoryManager(max_memories=2000)
        self.response_types = ["tell_story", "ask_question", "give_info"]
        self.max_context_length = 3
        self.current_conversation = []
        self.dataset_samples = []
        
        if os.path.exists(model_path):
            self.load_model()
    
    def train(self, data_file):
        X_text, y_text = self.preprocessor.load_dataset(data_file)
        self.preprocessor.build_vocab(X_text + y_text)
        print(f"Loaded {len(X_text)} sample pairs from dataset")
        
        valid_pairs = [(x, y) for x, y in zip(X_text, y_text) if len(x.strip()) > 5 and len(y.strip()) > 5]
        if len(valid_pairs) < len(X_text):
            print(f"Filtered out {len(X_text) - len(valid_pairs)} invalid samples")
        X_text, y_text = zip(*valid_pairs)
        self.dataset_samples = list(y_text)
        
        X_ids = [self.preprocessor.text_to_ids(x) for x in X_text]
        y_ids = [self.preprocessor.text_to_ids(y) for y in y_text]
        
        # Train Reasoner
        X_embed = np.array([np.mean(self.preprocessor.word_embeddings[x], axis=0) for x in X_ids])
        y_intents = np.array([
            0 if any(w in x.lower() for w in ["story", "tale", "once", "tell me a"]) else
            1 if any(w in x.lower() for w in ["how", "what", "why", "who"]) else
            2 for x in X_text
        ])
        y_one_hot = np.eye(3)[y_intents]
        
        self.reasoner = ReasoningEngine(self.preprocessor.vocab_size)
        self.reasoner.train(X_embed, y_one_hot)
        
        # Train Generator
        self.generator = Generator(self.preprocessor.vocab_size)
        self.generator.train(X_ids, y_ids)
        
        # Seed memory
        for x, y in zip(X_text, y_text):
            self.memory_manager.add_memory(x, y)
        
        self.save_model()
    
    def process_input(self, user_input, context=None):
        if context is None:
            context = []
        context.append({"role": "user", "content": user_input})
        self.current_conversation.append({"role": "user", "content": user_input})
        if len(context) > self.max_context_length * 2:
            context = context[-self.max_context_length * 2:]
        
        context_text = " ".join(item["content"] for item in context)
        input_ids = self.preprocessor.text_to_ids(context_text)
        input_embed = np.mean(self.preprocessor.word_embeddings[input_ids], axis=0).reshape(1, -1)
        
        # Check user memory first
        memories = self.memory_manager.get_relevant_memories(user_input)
        if memories and memories[0]["relevance"] > 0.8:
            response = memories[0]["response"]
        else:
            # Check web data
            web_entries = self.memory_manager.get_web_data(user_input)
            if web_entries and web_entries[0]["relevance"] > 0.7:
                response = f"From the web: {web_entries[0]['content']}"
            else:
                # Scrape the web if no good match
                web_content = self.scrape_web(user_input)
                if web_content:
                    response = f"Found online: {web_content}"
                    self.memory_manager.add_web_data(user_input, web_content, "https://www.google.com/search?q=" + user_input.replace(" ", "+"))
                else:
                    # Generate if all else fails
                    intent_probs = self.reasoner.forward(input_embed)
                    intent_idx = np.argmax(intent_probs)
                    response_type = self.response_types[intent_idx]
                    prompt_ids = self.preprocessor.text_to_ids(f"[{response_type}] {user_input}")
                    output_ids = self.generator.generate(prompt_ids)
                    response = self.preprocessor.ids_to_text(output_ids[len(prompt_ids):])
        
        # Store user-Kai interaction in memory
        self.memory_manager.add_memory(user_input, response)
        
        context.append({"role": "assistant", "content": response})
        self.current_conversation.append({"role": "assistant", "content": response})
        
        if self.memory_manager.check_retrain_needed():
            self._retrain_from_memory()
        
        return response, context
    
    def scrape_web(self, query):
        """Simple web scraper using Google search"""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Grab first snippet or paragraph-like content
            for div in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd'):
                text = div.get_text()
                if len(text) > 50:  # Decent length
                    return text[:500]  # Cap at 500 chars
            return None
        except Exception as e:
            print(f"Web scrape failed: {e}")
            return None
    
    def _retrain_from_memory(self):
        training_data = self.memory_manager.get_training_data()
        if not training_data:
            return
        
        X_text = [item["input"] for item in training_data]
        y_text = [item["output"] for item in training_data]
        
        X_ids = [self.preprocessor.text_to_ids(x) for x in X_text]
        y_ids = [self.preprocessor.text_to_ids(y) for y in y_text]
        
        self.generator.train(X_ids, y_ids, epochs=5)
        self.memory_manager.clear_short_term_memory()
        self.save_model()
    
    def save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'preprocessor': self.preprocessor,
                'generator': self.generator,
                'reasoner': self.reasoner,
                'response_types': self.response_types,
                'dataset_samples': self.dataset_samples
            }, f)
        self.memory_manager.save_memories()
    
    def load_model(self):
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        self.preprocessor = data['preprocessor']
        self.generator = data['generator']
        self.reasoner = data['reasoner']
        self.response_types = data['response_types']
        self.dataset_samples = data['dataset_samples']