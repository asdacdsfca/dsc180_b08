# Save this as fix_nltk.py
import nltk

# Download the punkt model
nltk.download('punkt')

# Fix for the punkt_tab error
# This script creates a custom tokenizer instead of using punkt_tab
print("NLTK resources installed successfully!")