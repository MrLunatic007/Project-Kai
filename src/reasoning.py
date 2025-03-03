import re
import numpy as np
from src.utils import cosine_similarity

class ReasoningEngine:
    def __init__(self):
        self.reasoning_templates = {
            "deduction": "Based on what you're asking about {concept}, {conclusion}",
            "induction": "From what I know about {concept}, {conclusion}",
            "analogy": "It's similar to {concept}, so {conclusion}",
            "causal": "Because you mentioned {cause}, {conclusion}",
            "conditional": "If you're interested in {condition}, then {conclusion}"
        }
        self.rules = [
            {"pattern": r"why", "reasoning": "causal", "weight": 0.8},
            {"pattern": r"how", "reasoning": "deduction", "weight": 0.7},
            {"pattern": r"what if", "reasoning": "conditional", "weight": 0.9},
            {"pattern": r"like|similar", "reasoning": "analogy", "weight": 0.8},
            {"pattern": r"because|since", "reasoning": "causal", "weight": 0.7},
            {"pattern": r"are you|how's", "reasoning": "deduction", "weight": 0.6},
        ]
    
    def detect_reasoning_type(self, query):
        max_weight = 0
        reasoning_type = "deduction"  # Default
        for rule in self.rules:
            if re.search(rule["pattern"], query.lower()):
                if rule["weight"] > max_weight:
                    max_weight = rule["weight"]
                    reasoning_type = rule["reasoning"]
        return reasoning_type if max_weight > 0.5 else None
    
    def extract_key_concepts(self, text):
        """Extract meaningful concepts from text more reliably"""
        words = text.lower().split()
        stopwords = {"the", "a", "an", "is", "are", "to", "in", "on", "at", "of", "for", "with", "by", "as", "that", "this"}
        
        # Extract noun phrases (simplified approach)
        keywords = []
        for word in words:
            if word not in stopwords and len(word) > 2:
                keywords.append(word)
        
        # If no keywords found, use the last non-stopword
        if not keywords:
            for word in reversed(words):
                if word not in stopwords and len(word) > 2:
                    return [word]
            return ["that topic"]  # Fallback
        
        # Return 1-2 most relevant keywords
        return keywords[:2]
    
    def enhance_response(self, query, base_response, context=None):
        """Enhance response with reasoning, ensuring coherence"""
        # Detect reasoning type from query
        reasoning_type = self.detect_reasoning_type(query)
        if not reasoning_type or not context:
            return base_response
        
        # Extract concepts from query and context
        query_concepts = self.extract_key_concepts(query)
        if not query_concepts:
            query_concepts = ["that"]
            
        context_text = " ".join(item["content"] for item in context if item["role"] == "user")
        context_concepts = self.extract_key_concepts(context_text) if context_text else query_concepts
        if not context_concepts:
            context_concepts = ["that"]
        
        # Handle special case for greetings/how are you
        if "how are you" in query.lower() or "are you" in query.lower() and "how" in query.lower():
            return "I'm doing great, thanks for asking! " + base_response
        
        # Apply appropriate reasoning template
        try:
            if reasoning_type == "deduction":
                return self.reasoning_templates["deduction"].format(
                    concept=query_concepts[0],
                    conclusion=base_response
                )
            elif reasoning_type == "causal":
                return self.reasoning_templates["causal"].format(
                    cause=query_concepts[0],
                    conclusion=base_response
                )
            elif reasoning_type == "conditional":
                return self.reasoning_templates["conditional"].format(
                    condition=query_concepts[0],
                    conclusion=base_response
                )
            elif reasoning_type == "analogy":
                return self.reasoning_templates["analogy"].format(
                    concept=query_concepts[0],
                    conclusion=base_response
                )
            else:  # induction
                return self.reasoning_templates["induction"].format(
                    concept=query_concepts[0],
                    conclusion=base_response
                )
        except Exception:
            # Fallback if template application fails
            return base_response