#!/bin/bash

echo "Running Reddit Scraper..."
cd scraping
python with_comment.py
cd ..  # Go back to root directory

echo "Running Basic Posts Filter..."
cd sentiment+filtering
python fix_nltk.py
python language_processing.py
python additional_tech_post_filter.py  # Ensure .py extension
cd ..  # Go back to root directory

echo "Running Basic Pre-trained Model Sentiment Filter..."
cd sentiment+filtering
python basic_sentiment.py
python additional_sentiment.py
cd ..  # Go back to root directory

echo "Running OpenAI Post Filter..."
cd sentiment+filtering  # Ensure correct path if agents_filtering.py is inside this folder
python agents_filtering.py
cd ..  # Go back to root directory

echo "Processing Complete."
