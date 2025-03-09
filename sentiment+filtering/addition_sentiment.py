import pandas as pd
import re
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class StockSentimentAnalyzer:
    def __init__(self, use_gpu=True):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pre-trained finance sentiment model
        self.model_name = "ProsusAI/finbert"  # Finance-specific BERT model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Company-specific context terms to include in analysis
        self.company_context = {
            
            # New companies
            'Broadcom': 'semiconductors, chips, networking, VMware, Symantec, Hock Tan',
            'Oracle': 'database, enterprise software, cloud, Java, Larry Ellison, Safra Catz',
            'Salesforce': 'CRM, cloud software, Slack, Tableau, Marc Benioff',
            'Cisco': 'networking, routers, switches, WebEx, Meraki, Chuck Robbins',
            'IBM': 'enterprise computing, cloud, AI, Watson, Red Hat, Arvind Krishna',
            'Adobe': 'creative software, Photoshop, PDF, Acrobat, Creative Cloud, Shantanu Narayen',
            'Qualcomm': 'mobile chips, Snapdragon, 5G, wireless, Cristiano Amon',
            'AMD': 'processors, CPUs, GPUs, servers, Ryzen, Radeon, Lisa Su',
            'Palantir': 'data analytics, big data, government contracts, Gotham, Foundry, Alex Karp'
        }
        
        # Confidence thresholds - higher as requested
        self.high_confidence = 0.75  # Threshold for high confidence classification
    
    def analyze_sentiment(self, content, mentioned_companies, score=1):
        """
        Analyze sentiment of content for mentioned companies using pre-trained model
        
        Returns:
        dict: Company to sentiment mapping with confidence scores
        """
        if not content or content == 'NaN' or pd.isna(content):
            return {}
            
        # Initialize results
        results = {}
        
        # No companies mentioned
        if not mentioned_companies or len(mentioned_companies) == 0:
            return {}
            
        # Convert string representation to list if needed
        if isinstance(mentioned_companies, str):
            # Handle string representation of list
            if mentioned_companies.startswith('[') and mentioned_companies.endswith(']'):
                mentioned_companies = eval(mentioned_companies)
            else:
                mentioned_companies = [mentioned_companies]
                
        # Process each mentioned company
        for company in mentioned_companies:
            # Skip if not a string
            if not isinstance(company, str):
                continue
                
            # Remove quotes and trim
            company = company.strip("'\" ")
            
            # Skip if empty
            if not company:
                continue
            
            # Check if the company is actually discussed in the content
            content_lower = str(content).lower()
            company_lower = company.lower()
            
            # Get related terms for the company
            company_terms = []
            if company in self.company_context:
                company_terms = [term.strip().lower() for term in self.company_context[company].split(',')]
            
            # Check if company or related terms are actually mentioned in content
            company_explicitly_mentioned = company_lower in content_lower
            related_terms_mentioned = any(term in content_lower for term in company_terms)
            
            # If company is not explicitly mentioned and no related terms, 
            # it might be a case where company is detected but post is about something else
            if not company_explicitly_mentioned and not related_terms_mentioned:
                # Mark as unclassified since the post may not really be about this company
                results[company] = {
                    'sentiment': "Unclassified",
                    'confidence': 0.0,
                    'score_weight': score,
                    'reason': "Company not explicitly discussed in post"
                }
                continue
                
            # Create company-specific context to help model focus
            if company in self.company_context:
                # Ask the model specific questions about the company
                augmented_content = (
                    f"Company: {company}. "
                    f"Context: {self.company_context[company]}. "
                    f"Post: {content}. "
                    f"Question: What is the sentiment specifically about {company} in this post?"
                )
            else:
                augmented_content = (
                    f"Company: {company}. "
                    f"Post: {content}. "
                    f"Question: What is the sentiment specifically about {company} in this post?"
                )
                
            # Prepare inputs for the model
            inputs = self.tokenizer(
                augmented_content,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            ).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()
            
            # FinBERT output format: [negative, neutral, positive]
            negative_score = probabilities[0]
            neutral_score = probabilities[1]
            positive_score = probabilities[2]
            
            # Check for competitor mentions which might confuse sentiment
            competitors = self.get_competitors(company)
            competitor_mentioned = any(comp.lower() in content_lower for comp in competitors)
            
            # If a competitor is mentioned, increase the confidence threshold
            confidence_threshold = self.high_confidence
            if competitor_mentioned:
                confidence_threshold = self.high_confidence + 0.1  # Even higher threshold
            
            # Calculate confidence as the max probability
            confidence = max(negative_score, neutral_score, positive_score)
                
            # Determine final sentiment
            if confidence >= confidence_threshold:
                if positive_score > negative_score and positive_score > neutral_score:
                    sentiment = "Bullish"
                elif negative_score > positive_score and negative_score > neutral_score:
                    sentiment = "Bearish"
                else:
                    sentiment = "Neutral"
            else:
                sentiment = "Unclassified"
                
            # Store result with confidence
            results[company] = {
                'sentiment': sentiment,
                'confidence': float(confidence),  # Convert from numpy to Python float
                'score_weight': score,
                'probabilities': {
                    'negative': float(negative_score),
                    'neutral': float(neutral_score),
                    'positive': float(positive_score)
                },
                'competitor_mentioned': competitor_mentioned
            }
            
        return results
        
    def get_competitors(self, company):
        """Return list of competitors for a given company"""
        competitors = {
            
            # New companies
            'Broadcom': ['Nvidia', 'Intel', 'Qualcomm', 'AMD', 'Marvel', 'Texas Instruments'],
            'Oracle': ['Microsoft', 'SAP', 'IBM', 'Salesforce', 'AWS', 'Google Cloud'],
            'Salesforce': ['Microsoft', 'Oracle', 'SAP', 'Adobe', 'Zoho', 'HubSpot'],
            'Cisco': ['Juniper', 'Arista', 'Huawei', 'HP Enterprise', 'Nokia'],
            'IBM': ['Microsoft', 'Amazon', 'Google', 'Oracle', 'SAP', 'Dell'],
            'Adobe': ['Canva', 'Figma', 'Corel', 'Microsoft', 'Affinity'],
            'Qualcomm': ['Broadcom', 'Intel', 'MediaTek', 'Samsung', 'Apple', 'Nvidia'],
            'AMD': ['Intel', 'Nvidia', 'Qualcomm', 'ARM', 'TSMC'],
            'Palantir': ['Snowflake', 'Databricks', 'Microsoft', 'IBM', 'Oracle', 'SAS']
        }
        
        return competitors.get(company, [])
    
    def analyze_dataframe(self, df):
        """
        Analyze all posts in a DataFrame
        
        Returns:
        DataFrame: Original DataFrame with added sentiment columns
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Add sentiment columns
        result_df['sentiment_analysis'] = None
        
        # Process each row
        for idx, row in result_df.iterrows():
            content = row.get('content', '')
            mentioned_companies = row.get('mentioned_companies', [])
            score = row.get('score', 1)
            
            sentiment_results = self.analyze_sentiment(content, mentioned_companies, score)
            result_df.at[idx, 'sentiment_analysis'] = sentiment_results
        
        # Add individual sentiment columns for all tracked companies
        all_companies = [
            'Broadcom', 'Oracle', 'Salesforce', 'Cisco', 'IBM', 'Adobe', 'Qualcomm', 'AMD', 'Palantir'
        ]
        
        for company in all_companies:
            col_name = f'{company}_sentiment'
            result_df[col_name] = result_df.apply(
                lambda row: row['sentiment_analysis'].get(company, {}).get('sentiment', '') 
                if isinstance(row['sentiment_analysis'], dict) else '',
                axis=1
            )
        
        return result_df
    
    def aggregate_daily_sentiment(self, df):
        """
        Aggregate sentiment by date for each company, weighted by score
        
        Returns:
        DataFrame: Daily sentiment summary for each company
        """
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Create empty results dataframe
        all_companies = [
            # Original companies
            # New companies
            'Broadcom', 'Oracle', 'Salesforce', 'Cisco', 'IBM', 'Adobe', 'Qualcomm', 'AMD', 'Palantir'
        ]
        results = []
        
        # Process each date
        for date in df['date'].unique():
            date_df = df[df['date'] == date]
            
            for company in all_companies:
                # Get all sentiment analyses for this company and date
                sentiments = []
                total_posts = 0
                
                for _, row in date_df.iterrows():
                    if isinstance(row['sentiment_analysis'], dict) and company in row['sentiment_analysis']:
                        total_posts += 1
                        sentiment_info = row['sentiment_analysis'][company]
                        if sentiment_info['sentiment'] != 'Unclassified':
                            sentiments.append({
                                'sentiment': sentiment_info['sentiment'],
                                'weight': sentiment_info['score_weight']
                            })
                
                # Calculate weighted sentiment
                if sentiments:
                    bullish_weight = sum(item['weight'] for item in sentiments if item['sentiment'] == 'Bullish')
                    bearish_weight = sum(item['weight'] for item in sentiments if item['sentiment'] == 'Bearish')
                    neutral_weight = sum(item['weight'] for item in sentiments if item['sentiment'] == 'Neutral')
                    
                    total_weight = bullish_weight + bearish_weight + neutral_weight
                    total_score_weight = sum(item['weight'] for item in sentiments)
                    
                    if total_weight > 0:
                        bullish_pct = bullish_weight / total_weight * 100
                        bearish_pct = bearish_weight / total_weight * 100
                        neutral_pct = neutral_weight / total_weight * 100
                        
                        # Calculate classified percentage
                        classified_pct = len(sentiments) / total_posts * 100 if total_posts > 0 else 0
                        
                        # Determine overall sentiment
                        if bullish_pct > bearish_pct and bullish_pct > neutral_pct:
                            overall = 'Bullish'
                        elif bearish_pct > bullish_pct and bearish_pct > neutral_pct:
                            overall = 'Bearish'
                        else:
                            overall = 'Neutral'
                            
                        results.append({
                            'date': date,
                            'company': company,
                            'bullish_pct': bullish_pct,
                            'bearish_pct': bearish_pct,
                            'neutral_pct': neutral_pct,
                            'overall_sentiment': overall,
                            'classified_posts': len(sentiments),
                            'total_posts': total_posts,
                            'classified_pct': classified_pct,
                            'total_score_weight': total_score_weight,
                            'avg_post_weight': total_score_weight / len(sentiments) if len(sentiments) > 0 else 0
                        })
                elif total_posts > 0:
                    # Add entry for company with mentions but no classified sentiment
                    results.append({
                        'date': date,
                        'company': company,
                        'bullish_pct': 0,
                        'bearish_pct': 0,
                        'neutral_pct': 0,
                        'overall_sentiment': 'Insufficient Data',
                        'classified_posts': 0,
                        'total_posts': total_posts,
                        'classified_pct': 0,
                        'total_score_weight': 0,
                        'avg_post_weight': 0
                    })
        
        # Convert to DataFrame
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=['date', 'company', 'bullish_pct', 'bearish_pct', 
                                        'neutral_pct', 'overall_sentiment', 'classified_posts',
                                        'total_posts', 'classified_pct', 'total_score_weight', 
                                        'avg_post_weight'])

# Example usage
if __name__ == "__main__":
    # Sample data (replace with actual data loading)
    data = pd.read_csv('high_confidence_posts_additional.csv')
    
    # Initialize analyzer (set use_gpu=False if no GPU available)
    analyzer = StockSentimentAnalyzer(use_gpu=True)
    
    # For single post testing
    sample_content = "Palantir's government contracts make it well-positioned for the growing AI demand in defense sector."
    sample_companies = ['Palantir']
    sample_sentiment = analyzer.analyze_sentiment(sample_content, sample_companies)
    print(f"Sample sentiment analysis: {sample_sentiment}")
    
    # Analyze sentiment for all posts
    print("Analyzing sentiment for all posts...")
    results = analyzer.analyze_dataframe(data)
    
    # Save results
    results.to_csv('sentiment_analysis_results.csv', index=False)
    
    # Aggregate daily sentiment
    daily_sentiment = analyzer.aggregate_daily_sentiment(results)
    daily_sentiment.to_csv('daily_sentiment_summary.csv', index=False)
    
    # Print summary of classification results
    # Count all classified sentiments for all companies
    all_companies = [
        # New companies
        'Broadcom', 'Oracle', 'Salesforce', 'Cisco', 'IBM', 'Adobe', 'Qualcomm', 'AMD', 'Palantir'
    ]
    
    classified = 0
    for company in all_companies:
        col_name = f'{company}_sentiment'
        if col_name in results.columns:
            classified += results[results[col_name] != ''].shape[0]
    
    print(f"Classified {classified} company mentions with high confidence threshold of {analyzer.high_confidence}")
    print(results[['content', 'mentioned_companies', 'sentiment_analysis']].head())