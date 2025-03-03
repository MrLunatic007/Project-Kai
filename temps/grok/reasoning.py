import re
import numpy as np
from src.utils import cosine_similarity

class ReasoningEngine:
    def __init__(self):
        self.reasoning_templates = {
            "deduction": "Since you asked about {concept}, here’s what I think: {conclusion}.",
            "induction": "Based on things like {examples}, I’d say {conclusion}.",
            "analogy": "It’s kind of like {analogy}, so {conclusion}.",
            "causal": "Because {cause}, it makes sense that {conclusion}.",
            "conditional": "If {condition}, then {conclusion}."
        }
        self.rules = [
            {"pattern": r"why", "reasoning": "causal", "weight": 0.8},
            {"pattern": r"how", "reasoning": "deduction", "weight": 0.7},
            {"pattern": r"what if", "reasoning": "conditional", "weight": 0.9},
            {"pattern": r"like|similar", "reasoning": "analogy", "weight": 0.8},
            {"pattern": r"because|since", "reasoning": "causal", "weight": 0.7},
            {"pattern": r"are you|how’s", "reasoning": "deduction", "weight": 0.6},
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
        words = text.lower().split()
        stopwords = {"the", "a", "an", "is", "are", "to", "in", "on", "at"}
        keywords = [word for word in words if word not in stopwords and len(word) > 1]
        return keywords[:2] if keywords else [text.split()[-1]]  # Last word as fallback
    
    def enhance_response(self, query, base_response, context=None):
        reasoning_type = self.detect_reasoning_type(query)
        if not reasoning_type or not context:
            return base_response
        
        query_concepts = self.extract_key_concepts(query)
        context_text = " ".join(item["content"] for item in context if item["role"] == "user")
        context_concepts = self.extract_key_concepts(context_text) if context_text else query_concepts
        
        if reasoning_type == "deduction":
            if "are you" in query.lower():
                return f"I’m doing great, thanks for asking! {base_response}"
            return self.reasoning_templates["deduction"].format(
                concept=query_concepts[0],
                conclusion=base_response
            )
        elif reasoning_type == "causal":
            return self.reasoning_templates["causal"].format(
                cause=f"you brought up {context_concepts[0] if context_concepts else 'this'}",
                conclusion=base_response
            )
        elif reasoning_type == "conditional":
            return self.reasoning_templates["conditional"].format(
                condition=f"you’re wondering about {query_concepts[0]}",
                conclusion=base_response
            )
        return base_response