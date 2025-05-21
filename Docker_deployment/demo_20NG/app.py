import os
import pandas as pd
import numpy as np
import joblib
import nltk
import gradio as gr
from gradio.themes import Soft
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

# Download stopwords
nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess a single text input, handling invalid entries"""
    if not isinstance(text, str) or pd.isna(text):
        raise ValueError("Input must be a valid string")
    
    # Remove excessive whitespace and clean text
    text = ' '.join(text.strip().split())
    # Remove special characters and numbers (optional, can be customized)
    text = ''.join(char for char in text if char.isalpha() or char.isspace())
    
    # Check if text is empty after cleaning
    if len(text.strip()) == 0:
        raise ValueError("Text is empty after preprocessing")
    
    return text

def classify_text(text, model_path, vectorizer_path=None):
    """
    Classify a single text input using the loaded model
    
    Args:
        text (str): The input text to classify
        model_path (str): Path to the joblib model file
        vectorizer_path (str, optional): Path to the TF-IDF vectorizer joblib file
    
    Returns:
        str: Predicted label
    """
    try:
        # Load the model
        model = joblib.load(model_path)
        print("✅ Model loaded successfully.")
        
        # Preprocess the input text
        processed_text = preprocess_text(text)
        print("✅ Text preprocessed successfully.")
        
        # Get the TF-IDF vectorizer
        if vectorizer_path and os.path.exists(vectorizer_path):
            # Load the pre-trained vectorizer if provided
            vectorizer = joblib.load(vectorizer_path)
            print("✅ Vectorizer loaded from file.")
        else:
            print("⚠️ No vectorizer file provided. Attempting to create compatible vectorizer...")
            
            # Since we don't have the original vectorizer, we need to recreate one with the same vocabulary
            # Load the 20 newsgroups dataset which was likely used for training
            newsgroups_train = fetch_20newsgroups(subset='train')
            
            # Create and fit a new vectorizer using the same dataset
            vectorizer = TfidfVectorizer(stop_words='english')
            vectorizer.fit(newsgroups_train.data)
            print(f"✅ Created vectorizer with {len(vectorizer.get_feature_names_out())} features.")
        
        # Vectorize the preprocessed text
        text_vectorized = vectorizer.transform([processed_text])
        print(f"✅ Text vectorized successfully with {text_vectorized.shape[1]} features.")
        
        # Predict using the model
        prediction = model.predict(text_vectorized)
        print("✅ Prediction completed.")
        
        return prediction[0]
    
    except Exception as e:
        print(f"❌ Error during classification: {str(e)}")
        return None

# Example usage

# Update these paths to your actual model and vectorizer files
model_path = './tf_idf_VotingClassifier.joblib'  
vectorizer_path = './tfidf_vectorizer.joblib'

def predict(input_text):
    label_mapping = {
        0: "Alternative: Atheism",
        1: "Computing: Graphics",
        2: "Computing: Microsoft Windows Miscellaneous",
        3: "Computing: IBM PC Hardware",
        4: "Computing: Macintosh Hardware",
        5: "Computing: X Window System",
        6: "Miscellaneous: Items for Sale",
        7: "Recreation: Automobiles",
        8: "Recreation: Motorcycles",
        9: "Recreation: Baseball",
        10: "Recreation: Hockey",
        11: "Science: Cryptography",
        12: "Science: Electronics",
        13: "Science: Medicine",
        14: "Science: Space Exploration",
        15: "Society: Christianity",
        16: "Talk: Gun Politics",
        17: "Talk: Middle East Politics",
        18: "Talk: General Politics",
        19: "Talk: Religion Miscellaneous"
    }
    # Classify the text
    result = classify_text(input_text, model_path, vectorizer_path)
    if result is not None:
        return f"{label_mapping[result]}"
    else:
        return "Classification failed. Please try again."

def clear_input():
    return ""  # Clears the input textbox

# Example inputs for users to try
examples = [
    "I just bought a new graphics card, any tips for installation?",
    "What are the ethical implications of space exploration?",
    "Looking for a used motorcycle in good condition."
]

# Custom CSS for styling
custom_css = """
/* Center the app content */
.gradio-container {
    max-width: 800px;
    margin: auto;
    padding: 20px;
}

/* Style the title */
h1 {
    color: #ffffff;
    text-align: center;
    font-family: 'Arial', sans-serif;
}

/* Style the description */
.description {
    text-align: center;
    font-size: 1.1em;
    color: #f0f0f0;
    margin-bottom: 20px;
}

/* Style the prediction output */
#prediction {
    background-color: #1f2937; /* Slightly darker, neutral background */
    color: #1a252f; /* Darker text for better contrast */
    padding: 15px;
    border-radius: 8px;
    font-weight: bold;
    border: 1px solid #b0c4de; /* Subtle border for definition */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
}

/* Style buttons */
button {
    border-radius: 8px;
    font-weight: bold;
}
"""

# Create the Gradio app with Blocks for better layout control
with gr.Blocks(theme=Soft(), css=custom_css) as app:
    gr.Markdown(
        """
        # Topic Classification App
        <div class="description">
        Enter text to classify it into one of 20 categories, such as technology, recreation, or politics.  
        Try the example inputs below or type your own text!
        </div>
        """
    )
    
    with gr.Row():
        input_text = gr.Textbox(
            label="Enter Your Text",
            placeholder="Type your text here (e.g., 'What are the benefits of electric cars?')",
            lines=5,
            max_lines=10
        )
    
    with gr.Row():
        submit_button = gr.Button("Classify", variant="primary")
        clear_button = gr.Button("Clear", variant="secondary")
    
    output = gr.Textbox(label="Prediction", elem_id="prediction", lines=3)
    
    gr.Examples(
        examples=examples,
        inputs=input_text,
        outputs=output,
        fn=predict,
        cache_examples=False,
        label="Try These Examples"
    )
    
    # Connect buttons to functions
    submit_button.click(fn=predict, inputs=input_text, outputs=output)
    clear_button.click(fn=clear_input, outputs=input_text)

# Launch the app
app.launch(server_name="0.0.0.0", server_port=7860)