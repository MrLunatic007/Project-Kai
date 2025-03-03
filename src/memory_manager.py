import numpy as np
import os
import json
import time
from datetime import datetime

class MemoryManager:
    """Manages personalized memory for the assistant"""
    
    def __init__(self, memory_path="data/memory", max_memories=2000, 
                 retrain_threshold=100, importance_threshold=0.5):
        self.memory_path = memory_path
        self.max_memories = max_memories
        self.retrain_threshold = retrain_threshold
        self.importance_threshold = importance_threshold
        
        # Create memory directory if it doesn't exist
        os.makedirs(memory_path, exist_ok=True)
        
        # Memory stores
        self.short_term_memory = []  # Recent interactions
        self.long_term_memory = []   # Important persistent memories
        self.training_needed = False
        
        # Load existing memories
        self._load_memories()
    
    def _load_memories(self):
        """Load memories from disk"""
        try:
            ltm_path = os.path.join(self.memory_path, "long_term_memory.json")
            stm_path = os.path.join(self.memory_path, "short_term_memory.json")
            
            if os.path.exists(ltm_path):
                with open(ltm_path, 'r') as f:
                    self.long_term_memory = json.load(f)
                print(f"Loaded {len(self.long_term_memory)} long-term memories")
            
            if os.path.exists(stm_path):
                with open(stm_path, 'r') as f:
                    self.short_term_memory = json.load(f)
                print(f"Loaded {len(self.short_term_memory)} short-term memories")
        except Exception as e:
            print(f"Error loading memories: {e}")
            # Initialize empty memories if loading fails
            self.short_term_memory = []
            self.long_term_memory = []
    
    def save_memories(self):
        """Save memories to disk"""
        try:
            ltm_path = os.path.join(self.memory_path, "long_term_memory.json")
            stm_path = os.path.join(self.memory_path, "short_term_memory.json")
            
            with open(ltm_path, 'w') as f:
                json.dump(self.long_term_memory, f)
            
            with open(stm_path, 'w') as f:
                json.dump(self.short_term_memory, f)
                
            print(f"Saved {len(self.long_term_memory)} long-term and {len(self.short_term_memory)} short-term memories")
        except Exception as e:
            print(f"Error saving memories: {e}")
    
    def add_memory(self, user_input, response, importance=None):
        """Add a new memory"""
        # Calculate importance if not provided
        if importance is None:
            importance = self._calculate_importance(user_input, response)
        
        timestamp = datetime.now().isoformat()
        
        memory = {
            "user_input": user_input,
            "response": response,
            "timestamp": timestamp,
            "importance": importance,
            "access_count": 0
        }
        
        # Add to short-term memory
        self.short_term_memory.append(memory)
        
        # If importance is high, add to long-term memory too
        if importance >= self.importance_threshold:
            self.long_term_memory.append(memory)
        
        # Check if we need to consolidate or retrain
        self._check_memory_limits()
        
        # Update training flag if needed
        if len(self.short_term_memory) >= self.retrain_threshold:
            self.training_needed = True
    
    def _calculate_importance(self, user_input, response):
        """Calculate importance score for a memory"""
        # Simplified importance calculation
        # Real implementation could use NLP features, sentiment, etc.
        importance = 0.5  # Default medium importance
        
        # Longer exchanges might be more important
        if len(user_input) > 100 or len(response) > 100:
            importance += 0.1
            
        # Check for question words that might indicate learning
        learning_indicators = ["how", "why", "what is", "explain", "teach"]
        if any(indicator in user_input.lower() for indicator in learning_indicators):
            importance += 0.2
            
        # Check for personalization indicators
        personal_indicators = ["i am", "my name", "i like", "i want", "i need", "i prefer"]
        if any(indicator in user_input.lower() for indicator in personal_indicators):
            importance += 0.3
            
        # Cap importance at 1.0
        return min(1.0, importance)
    
    def _check_memory_limits(self):
        """Check if memory limits have been reached and manage accordingly"""
        # Check short-term memory
        if len(self.short_term_memory) > self.max_memories:
            self._consolidate_memories()
            
        # Check long-term memory
        if len(self.long_term_memory) > self.max_memories:
            self._prune_long_term_memory()
    
    def _consolidate_memories(self):
        """Consolidate short-term memories"""
        # Sort by importance
        self.short_term_memory.sort(key=lambda x: x["importance"], reverse=True)
        
        # Keep important memories
        keep_count = int(self.max_memories * 0.3)  # Keep 30% of max capacity
        important_memories = self.short_term_memory[:keep_count]
        
        # Move some to long-term memory if not already there
        for memory in important_memories:
            if memory["importance"] >= self.importance_threshold:
                # Check if this memory is already in long-term memory
                if not any(ltm["user_input"] == memory["user_input"] and 
                           ltm["response"] == memory["response"] 
                           for ltm in self.long_term_memory):
                    self.long_term_memory.append(memory)
        
        # Reset short-term memory
        self.short_term_memory = important_memories
        print(f"Consolidated memories: kept {len(important_memories)} out of {self.max_memories}")
    
    def _prune_long_term_memory(self):
        """Remove less important long-term memories"""
        # Sort by a combination of importance, recency, and access count
        def memory_value(memory):
            # Importance is the primary factor
            value = memory["importance"] * 0.6
            
            # Recency (higher for newer memories)
            days_old = (datetime.now() - datetime.fromisoformat(memory["timestamp"])).days + 1
            recency_score = 1.0 / (days_old)
            value += recency_score * 0.2
            
            # Access frequency
            value += min(1.0, memory["access_count"] / 10) * 0.2
            
            return value
            
        self.long_term_memory.sort(key=memory_value, reverse=True)
        
        # Keep the top memories
        keep_count = int(self.max_memories * 0.8)  # Keep 80% of max capacity
        self.long_term_memory = self.long_term_memory[:keep_count]
        print(f"Pruned long-term memory: kept {keep_count} memories")
    
    def get_relevant_memories(self, user_input, max_results=5):
        """Retrieve memories relevant to the current input"""
        relevant_memories = []
        
        # Search in both short and long-term memory
        all_memories = self.short_term_memory + self.long_term_memory
        
        # Calculate relevance scores
        for memory in all_memories:
            relevance = self._calculate_relevance(user_input, memory["user_input"])
            if relevance > 0.3:  # Minimum relevance threshold
                memory["relevance"] = relevance
                relevant_memories.append(memory)
                
                # Increment access count for this memory
                memory["access_count"] += 1
        
        # Sort by relevance and return top results
        relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)
        return relevant_memories[:max_results]
    
    def _calculate_relevance(self, query, stored_input):
        """Calculate relevance between current query and stored memory"""
        # Simple word overlap for relevance
        query_words = set(query.lower().split())
        stored_words = set(stored_input.lower().split())
        
        if not query_words or not stored_words:
            return 0
            
        overlap = query_words.intersection(stored_words)
        
        # Relevance based on word overlap percentage
        relevance = len(overlap) / max(len(query_words), len(stored_words))
        
        return relevance
    
    def get_training_data(self):
        """Get data for retraining"""
        # Combine memories from both stores, avoiding duplicates
        training_data = []
        seen = set()
        
        for memory in self.short_term_memory + self.long_term_memory:
            key = (memory["user_input"], memory["response"])
            if key not in seen:
                seen.add(key)
                training_data.append({
                    "input": memory["user_input"],
                    "output": memory["response"]
                })
        
        # Reset training needed flag
        self.training_needed = False
        
        return training_data
    
    def check_retrain_needed(self):
        """Check if retraining is needed"""
        return self.training_needed
    
    def clear_short_term_memory(self):
        """Clear short-term memory after training"""
        # Save important memories to long-term first
        for memory in self.short_term_memory:
            if memory["importance"] >= self.importance_threshold:
                if not any(ltm["user_input"] == memory["user_input"] and 
                           ltm["response"] == memory["response"] 
                           for ltm in self.long_term_memory):
                    self.long_term_memory.append(memory)
        
        # Clear short-term memory
        self.short_term_memory = []
        print("Short-term memory cleared")