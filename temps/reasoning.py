import re
import numpy as np
from src.utils import cosine_similarity

class ReasoningEngine:
    """Engine for enhancing responses with reasoning capabilities"""
    
    def __init__(self):
        # Define reasoning techniques
        self.reasoning_templates = {
            "deduction": "Based on {premise}, we can conclude that {conclusion}.",
            "induction": "From observations like {examples}, we might infer {conclusion}.",
            "analogy": "This is similar to {analogy}, so {conclusion}.",
            "causal": "Since {cause} typically leads to {effect}, {conclusion}.",
            "conditional": "If {condition}, then {conclusion}."
        }
        
        # Rule patterns for basic reasoning
        self.rules = [
            # Question patterns
            {"pattern": r"why", "reasoning": "causal", "weight": 0.8},
            {"pattern": r"how", "reasoning": "deduction", "weight": 0.7},
            {"pattern": r"what if", "reasoning": "conditional", "weight": 0.9},
            {"pattern": r"compare", "reasoning": "analogy", "weight": 0.6},
            {"pattern": r"similar", "reasoning": "analogy", "weight": 0.8},
            
            # Topic patterns
            {"pattern": r"weather|temperature|climate", "reasoning": "induction", "weight": 0.6},
            {"pattern": r"math|calculate|compute", "reasoning": "deduction", "weight": 0.9},
            {"pattern": r"history|past|happened", "reasoning": "causal", "weight": 0.7},
            {"pattern": r"future|predict|forecast", "reasoning": "induction", "weight": 0.8},
            {"pattern": r"relationship|connection", "reasoning": "analogy", "weight": 0.7}
        ]
    
    def detect_reasoning_type(self, query):
        """Detect what type of reasoning would be appropriate"""
        max_weight = 0
        reasoning_type = "deduction"  # Default
        
        for rule in self.rules:
            if re.search(rule["pattern"], query.lower()):
                if rule["weight"] > max_weight:
                    max_weight = rule["weight"]
                    reasoning_type = rule["reasoning"]
        
        return reasoning_type if max_weight > 0.5 else None
    
    def extract_key_concepts(self, text):
        """Extract key concepts from text"""
        # Simple keyword extraction (could be improved)
        words = text.lower().split()
        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "in", "on", "at", "by", "for"}
        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        
        # Return top 3 keywords or fewer if not available
        return keywords[:3]
    
    def formulate_reasoning(self, query, response, reasoning_type):
        """Formulate reasoning based on the detected type"""
        if not reasoning_type:
            return response
        
        # Extract concepts from query and response
        query_concepts = self.extract_key_concepts(query)
        response_concepts = self.extract_key_concepts(response)
        
        # Create reasoning components
        reasoning_components = {}
        
        if reasoning_type == "deduction":
            reasoning_components = {
                "premise": f"you asked about {' and '.join(query_concepts)}",
                "conclusion": response
            }
        elif reasoning_type == "induction":
            reasoning_components = {
                "examples": f"your question about {' and '.join(query_concepts)}",
                "conclusion": response
            }
        elif reasoning_type == "analogy":
            reasoning_components = {
                "analogy": f"other situations involving {' or '.join(query_concepts)}",
                "conclusion": response
            }
        elif reasoning_type == "causal":
            reasoning_components = {
                "cause": f"your interest in {' and '.join(query_concepts)}",
                "effect": f"a focus on {' and '.join(response_concepts) if response_concepts else 'these topics'}",
                "conclusion": response
            }
        elif reasoning_type == "conditional":
            reasoning_components = {
                "condition": f"we consider {' and '.join(query_concepts)}",
                "conclusion": response
            }
        
        # Format the reasoning template
        reasoning_text = self.reasoning_templates[reasoning_type].format(**reasoning_components)
        
        return reasoning_text
    
    def enhance_response(self, query, base_response, context=None):
        """Enhance a response with reasoning"""
        # Detect what type of reasoning would be appropriate
        reasoning_type = self.detect_reasoning_type(query)
        
        # If no specific reasoning needed, return the base response
        if not reasoning_type:
            return base_response
        
        # Apply reasoning to formulate a more thoughtful response
        enhanced_response = self.formulate_reasoning(query, base_response, reasoning_type)
        
        return enhanced_response