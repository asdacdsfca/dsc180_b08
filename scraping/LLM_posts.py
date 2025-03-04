import pandas as pd
import openai
from openai import OpenAI
import time
from datetime import datetime

# Configure OpenAI API
client = OpenAI(api_key=OPENAI_API_KEY)

def analyze_post_content(title, content):
    """
    Analyze post content using OpenAI API to determine relevance
    """
    system_prompt = """You are a financial and technology content analyzer. 
    Evaluate the provided Reddit post for relevant information about stock market 
    analysis, technology industry developments, or substantive market insights for
    those companies in S&P500. Respond with either 'RELEVANT' or 'NOT RELEVANT'.
    If relevant than classify the sentiment of that post as 'Bullish', 
    'Bearish" or 'Neutral'"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Title: {title}\nContent: {content}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        time.sleep(20)  # Back off on error
        return None

def process_reddit_data(input_file, output_file):
    """
    Process Reddit data from CSV and filter using OpenAI
    """
    df = pd.read_csv(input_file)
    filtered_posts = []
    
    print(f"Processing {len(df)} posts...")
    
    for index, row in df.iterrows():
        if index % 10 == 0:
            print(f"Processed {index} posts...")
            
        analysis = analyze_post_content(row['title'], row['content'])
        
        if analysis and 'RELEVANT' in analysis.upper():
            filtered_posts.append({
                'date': row['date'],
                'title': row['title'],
                'content': row['content'],
                'url': row['url'],
                'score': row['score'],
                'ai_analysis': analysis
            })
        
        time.sleep(0.5)  # Rate limiting
    
    filtered_df = pd.DataFrame(filtered_posts)
    filtered_df.to_csv(output_file, index=False)
    
    print(f"\nAnalysis complete. Found {len(filtered_posts)} relevant posts.")
    return filtered_df

if __name__ == "__main__":
    input_file = 'reddit_posts.csv'
    output_file = 'filtered_reddit_posts.csv'
    
    filtered_data = process_reddit_data(input_file, output_file)