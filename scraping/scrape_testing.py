import praw
from datetime import datetime
import pandas as pd
import time
import re
from langdetect import detect, LangDetectException

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
    read_only=True
)

# List of S&P 500 technology sector companies and their tickers
tech_keywords = [
    'apple', 'aapl', 'microsoft', 'msft', 'nvidia', 'nvda', 'broadcom', 'avgo',
    'oracle', 'orcl', 'cisco', 'csco', 'salesforce', 'crm', 'adobe', 'adbe',
    'intel', 'intc', 'ibm', 'qcom', 'amd', 'txn', 'amat', 'servicenow', 'now',
    'uber', 'shopify', 'shop', 'mongodb', 'mdb', 'synopsys', 'snps', 'cadence',
    'cdns', 'intuit', 'intu', 'autodesk', 'adsk', 'adi', 'micron', 'mu', 'anet',
    'lrcx', 'marvell', 'mrvl', 'fortinet', 'ftnt', 'nxpi', 'panw', 'atlassian',
    'team', 'msi', 'rop', 'wday', 'ttd', 'cloudflare', 'net', 'on', 'gfs',
    'mchp', 'crwd', 'zscaler', 'zs', 'palantir', 'pltr', 'fslr', 'akamai', 'akam',
    'leidos', 'ldos', 'corning', 'glw', 'jnpr', 'ter', 'tyl', 'ffiv', 'br', 'cdw',
    'ntap', 'jkhy', 'vrsn', 'swks', 'it', 'stx', 'hp', 'hpe', 'dxc', 'xrx', 'ctsh',
    'pypl', 'square', 'sq', 'docu', 'zm', 'okta', 'twlo', 'estc', 'splk', 'mstr',
    'dbx', 'rng'
]

# Compile regex pattern for keyword matching
pattern = re.compile(r'\b(' + '|'.join(re.escape(keyword) for keyword in tech_keywords) + r')\b', flags=re.IGNORECASE)

def clean_text(text):
    """Clean and standardize text content"""
    if not text:
        return ""
    # Remove special characters and emojis
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_reddit_posts(subreddit_name, post_limit):
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    
    for post in subreddit.new(limit=post_limit):
        try:
            cleaned_title = clean_text(post.title)
            cleaned_content = clean_text(post.selftext)
            combined_text = f"{cleaned_title} {cleaned_content}"
            
            # Check if the post is in English
            try:
                if detect(combined_text) != 'en':
                    print(f"Skipping non-English post: {post.title[:50]}...")
                    continue
            except LangDetectException:
                print(f"Language detection failed for post: {post.title[:50]}...")
                continue
            
            # Check if the post mentions any tech keywords
            if not pattern.search(combined_text):
                print(f"Skipping post without tech keyword: {post.title[:50]}...")
                continue
            
            # Collect post data if it passes both filters
            post_data = {
                'date': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                'title': cleaned_title,
                'content': cleaned_content,
                'url': post.url,
                'score': post.score
            }
            posts_data.append(post_data)
            print(f"Retrieved post: {post.title[:50]}...")
            
        except Exception as e:
            print(f"Error processing post: {str(e)}")
            continue
        
        time.sleep(0.1)
    
    df = pd.DataFrame(posts_data)
    print(f"Total posts retrieved after filtering: {len(df)}")
    return df
# Usage
subreddit_name = "wallstreetbets"  # or other subreddits like "investing", "stocks"
df = scrape_reddit_posts(subreddit_name, post_limit=500000)
# start_date = datetime(2020, 1, 1)
# end_date = datetime(2025, 2, 1)
# subreddit_name = "wallstreetbets"  # Can use multiple subreddits
# df = scrape_reddit_posts(subreddit_name, start_date, end_date)
df.to_csv('wallstreetbets_reddit.csv', index=False)