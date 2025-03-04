#!/bin/bash
echo "Running Reddit Scraper..."
python reddit_scraper.py

echo "Running OpenAI Post Filter..."
python openai_filter.py

echo "Processing Complete."
