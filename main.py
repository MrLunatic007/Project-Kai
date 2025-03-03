from src.assistant import PersonalAssistant
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Kai Personal Assistant')
    parser.add_argument('--data', type=str, default="data/datasets/wiki_curated.csv", 
                        help='Path to training data')
    parser.add_argument('--model', type=str, default="models/kai_model.pkl",
                        help='Path to save/load model')
    parser.add_argument('--train', action='store_true', 
                        help='Train the model before starting')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(argsMODEL_PATH), exist_ok=True)
    
    assistant = PersonalAssistant(model_path=args.model)
    
    if args.train or not os.path.exists(args.model):
        print(f"Training Kai on {args.data}...")
        assistant.train(args.data)
        print(f"Model saved to {args.model}")
    else:
        assistant.load_model()
        print("Model loaded successfully")
    
    print("Kai Assistant is ready! (Type 'exit' to quit)")
    
    context = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response, context = assistant.process_input(user_input, context)
        print(f"Kai: {response}")