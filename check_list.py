import pickle
import os

print("Checking model files...")

# Check if files exist
print(f"model.pkl exists: {os.path.exists('model.pkl')}")
print(f"vectorizer.pkl exists: {os.path.exists('vectorizer.pkl')}")

if os.path.exists('model.pkl'):
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")

        # Check if model is fitted
        if hasattr(model, 'classes_'):
            print("✅ Model IS fitted")
            print(f"Model classes: {model.classes_}")
        else:
            print("❌ Model is NOT fitted")

    except Exception as e:
        print(f"Error loading model: {e}")

if os.path.exists('vectorizer.pkl'):
    try:
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("Vectorizer loaded successfully!")

        # Check if vectorizer is fitted
        if hasattr(vectorizer, 'vocabulary_'):
            print("✅ Vectorizer IS fitted")
            print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
        else:
            print("❌ Vectorizer is NOT fitted")

    except Exception as e:
        print(f"Error loading vectorizer: {e}")