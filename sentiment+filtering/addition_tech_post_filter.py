import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForTokenClassification
from sklearn.ensemble import RandomForestClassifier
import re
from tqdm import tqdm

class Mag7Dataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

class BERTMag7Classifier:
    def __init__(self, use_gpu=True):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load BERT
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model = self.bert_model.to(self.device)
        
        # Set to evaluation mode
        self.bert_model.eval()
        
        # Load NER model for organization detection
        print("Loading NER model for organization detection...")
        self.ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.ner_model = self.ner_model.to(self.device)
        self.ner_model.eval()
        
        # Company data for rules-based filtering
        self.company_data = {
            'Apple': ['apple', 'aapl', 'iphone', 'ipad', 'macbook', 'tim cook', 'ios'],
            'Microsoft': ['microsoft', 'msft', 'windows', 'azure', 'nadella', 'office'],
            'Nvidia': ['nvidia', 'nvda', 'gpu', 'jensen huang', 'cuda', 'geforce'],
            # New companies
            'Broadcom': ['broadcom', 'avgo', 'tan hock', 'hock tan', 'symantec', 'vmware'],
            'Oracle': ['oracle', 'orcl', 'larry ellison', 'safra catz', 'java', 'mysql', 'netsuite'],
            'Salesforce': ['salesforce', 'crm', 'benioff', 'marc benioff', 'tableau', 'slack'],
            'Cisco': ['cisco', 'csco', 'chuck robbins', 'webex', 'meraki'],
            'IBM': ['ibm', 'international business machines', 'arvind krishna', 'watson', 'red hat'],
            'Adobe': ['adobe', 'adbe', 'photoshop', 'illustrator', 'acrobat', 'shantanu narayen'],
            'Qualcomm': ['qualcomm', 'qcom', 'snapdragon', 'cristiano amon'],
            'AMD': ['amd', 'advanced micro devices', 'lisa su', 'ryzen', 'radeon', 'epyc'],
            'Palantir': ['palantir', 'pltr', 'alex karp', 'karp', 'gotham', 'foundry', 'apollo', 'data analytics']
        }
        
        # Company aliases and alternate names
        self.company_aliases = {
            'Apple': ['apple inc', 'apple corporation', 'cupertino giant', 'iphone maker'],
            'Microsoft': ['microsoft corporation', 'redmond giant', 'windows maker'],
            'Nvidia': ['nvidia corporation', 'chip maker', 'gpu giant'],
            # New companies
            'Broadcom': ['broadcom inc', 'broadcom limited', 'semiconductor company'],
            'Oracle': ['oracle corporation', 'database giant', 'enterprise software company'],
            'Salesforce': ['salesforce.com', 'salesforce inc', 'crm giant', 'cloud software company'],
            'Cisco': ['cisco systems', 'networking giant', 'router company'],
            'IBM': ['international business machines corporation', 'big blue'],
            'Adobe': ['adobe inc', 'adobe systems', 'creative software company'],
            'Qualcomm': ['qualcomm incorporated', 'mobile chip maker', 'wireless technology company'],
            'AMD': ['advanced micro devices inc', 'cpu maker', 'semiconductor company'],
            'Palantir': ['palantir technologies', 'palantir technologies inc', 'big data company', 'data analytics firm']
        }
        
        # All terms for quick filtering
        self.all_terms = []
        for terms in self.company_data.values():
            self.all_terms.extend(terms)
            
        # Classifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def preprocess_text(self, text):
        """Simple text preprocessing"""
        if not text or pd.isna(text):
            return ""
        # Convert to lowercase
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.strip()
        
    def contains_mag7_term(self, text):
        """Check if text contains any Mag7 term"""
        text = self.preprocess_text(text)
        return any(term in text for term in self.all_terms)
    
    def extract_organizations_ner(self, text):
        """Extract organization entities using NER model"""
        if not text or len(text) < 5:
            return []
            
        # Keep original case for NER
        try:
            # Tokenize text
            inputs = self.ner_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.ner_model(**inputs)
                
            predictions = torch.argmax(outputs.logits, dim=2)
            
            # Convert predictions to entity names
            tokens = self.ner_tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            token_predictions = [self.ner_model.config.id2label[t.item()] for t in predictions[0]]
            
            # Extract organization entities
            organizations = []
            current_org = []
            
            for token, prediction in zip(tokens, token_predictions):
                if prediction == "B-ORG":  # Beginning of organization
                    if current_org:
                        # Add previous org to list
                        organizations.append("".join(current_org).replace("##", ""))
                        current_org = []
                    current_org.append(token)
                elif prediction == "I-ORG":  # Inside organization
                    current_org.append(token)
                elif current_org:  # End of organization
                    organizations.append("".join(current_org).replace("##", ""))
                    current_org = []
                    
            # Add the last organization if it exists
            if current_org:
                organizations.append("".join(current_org).replace("##", ""))
                
            # Clean up organizations
            clean_orgs = []
            for org in organizations:
                # Remove special tokens
                if org not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                    # Clean up the organization name
                    clean_org = org.replace('[CLS]', '').replace('[SEP]', '').strip()
                    if clean_org:
                        clean_orgs.append(clean_org)
                        
            return clean_orgs
            
        except Exception as e:
            print(f"Error in NER processing: {e}")
            return []
    
    def map_organization_to_mag7(self, org_name):
        """Map detected organization to Mag7 company if possible"""
        org_name = org_name.lower()
        
        # Direct match with company names
        for company in self.company_data.keys():
            if company.lower() in org_name:
                return company
                
        # Check company terms
        for company, terms in self.company_data.items():
            if any(term in org_name for term in terms):
                return company
                
        # Check aliases
        for company, aliases in self.company_aliases.items():
            if any(alias in org_name for alias in aliases):
                return company
                
        return None
    
    def get_mentioned_companies(self, text):
        """Enhanced company detection with NER and word boundaries"""
        if not text or pd.isna(text):
            return []
            
        mentioned = set()
        
        # 1. Rule-based detection with word boundaries
        text_lower = self.preprocess_text(text)
        for company, terms in self.company_data.items():
            for term in terms:
                if len(term) < 4:
                    # Use word boundary check for short terms
                    if re.search(r'\b{}\b'.format(re.escape(term)), text_lower):
                        mentioned.add(company)
                        break
                elif term in text_lower:
                    mentioned.add(company)
                    break
        
        # 2. Check aliases
        for company, aliases in self.company_aliases.items():
            if any(alias in text_lower for alias in aliases):
                mentioned.add(company)
        
        # 3. NER-based detection
        try:
            # Get original text for NER (not lowercase)
            organizations = self.extract_organizations_ner(text)
            
            # Map organizations to Mag7 companies
            for org in organizations:
                mag7_company = self.map_organization_to_mag7(org)
                if mag7_company:
                    mentioned.add(mag7_company)
        except Exception as e:
            print(f"Error in NER company detection: {e}")
        
        # 4. Sector relationships
        energy_terms = ['oil', 'gas', 'energy', 'chevron', 'exxon', 'shell']
        chip_terms = ['semiconductor', 'chip', 'tsmc', 'intel']
        cloud_terms = ['cloud computing', 'saas', 'paas', 'iaas', 'data center']
        data_analytics_terms = ['big data', 'data analytics', 'data mining', 'predictive analytics']
        
        # Energy sector impacts Tesla when EV is mentioned
        if ('Tesla' not in mentioned and 
            any(term in text_lower for term in energy_terms) and
            any(ev_term in text_lower for ev_term in ['ev', 'electric vehicle', 'electric car'])):
            mentioned.add('Tesla')
            
        # Semiconductor sector impacts Nvidia, AMD, Broadcom, and Qualcomm when AI is mentioned
        if any(term in text_lower for term in chip_terms) and any(ai_term in text_lower for ai_term in ['ai', 'artificial intelligence', 'machine learning']):
            if 'Nvidia' not in mentioned:
                mentioned.add('Nvidia')
            if 'AMD' not in mentioned:
                mentioned.add('AMD')
            if 'Broadcom' not in mentioned:
                mentioned.add('Broadcom')
            if 'Qualcomm' not in mentioned:
                mentioned.add('Qualcomm')
                
        # Cloud computing impacts IBM, Microsoft, Amazon, Oracle, and Salesforce
        if any(term in text_lower for term in cloud_terms):
            for cloud_company in ['IBM', 'Microsoft', 'Amazon', 'Oracle', 'Salesforce']:
                if cloud_company not in mentioned:
                    mentioned.add(cloud_company)
                    
        # Data analytics impacts Palantir when relevant terms are mentioned
        if any(term in text_lower for term in data_analytics_terms) and 'Palantir' not in mentioned:
            mentioned.add('Palantir')
            
        return list(mentioned)
        
    def extract_bert_features(self, texts, batch_size=16):
        """Extract BERT features from texts"""
        # Create dataset
        dataset = Mag7Dataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # Extract features
        all_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting BERT features"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get [CLS] embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
                
        # Concatenate
        return np.vstack(all_embeddings)
        
    def generate_synthetic_training_data(self):
        """Generate synthetic training data"""
        training_texts = []
        labels = []
        
        # Finance terms
        finance_terms = ['stock', 'share', 'price', 'market', 'invest', 'trading', 'earnings']
        
        # Generate positive examples
        for company, terms in self.company_data.items():
            for term in terms:
                # Basic mention
                text = f"Discussion about {term} in the stock market"
                training_texts.append(text)
                labels.append(1)
                
                # With financial term
                for finance_term in finance_terms[:3]:  # Limit for brevity
                    text = f"{term} {finance_term} is looking interesting"
                    training_texts.append(text)
                    labels.append(1)
                    
                    # More complex examples
                    text = f"What do you think about the {finance_term} of {term}? The market seems bullish."
                    training_texts.append(text)
                    labels.append(1)
        
        # Generate negative examples
        topics = ['weather', 'politics', 'food', 'sports', 'travel', 'health', 'movies', 'music', 'books']
        for topic in topics:
            for i in range(5):  # Several examples per topic
                text = f"{topic} discussion thread {i}. What do you think about recent developments?"
                training_texts.append(text)
                labels.append(0)
                
        return training_texts, labels
        
    def train_model(self):
        """Train the classifier on synthetic data"""
        # Generate training data
        texts, labels = self.generate_synthetic_training_data()
        
        # Extract BERT features
        X = self.extract_bert_features(texts)
        
        # Train classifier
        self.classifier.fit(X, labels)
        
        return self.classifier
        
    def predict(self, texts):
        """Predict on new texts"""
        # Extract BERT features
        X = self.extract_bert_features(texts)
        
        # Predict
        return self.classifier.predict(X), self.classifier.predict_proba(X)[:, 1]
        
    def filter_mag7_posts(self, df, confidence_threshold=0.6, borderline_margin=0.1):
        """
        Filter DataFrame for Mag7-related posts with NER-enhanced company detection
        
        Parameters:
        df (pandas.DataFrame): DataFrame containing posts
        confidence_threshold (float): Threshold for confident classification
        borderline_margin (float): Margin below threshold to consider borderline
        
        Returns:
        pandas.DataFrame: Filtered DataFrame with classification and confidence info
        """
        # Quick filter to reduce processing
        df['combined_text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
        df['has_mag7_term'] = df['combined_text'].apply(self.contains_mag7_term)
        
        filtered_df = df[df['has_mag7_term']].copy()
        
        if len(filtered_df) == 0:
            print("No potentially relevant posts found.")
            return pd.DataFrame(columns=df.columns)
            
        print(f"Quick filter found {len(filtered_df)} potential posts out of {len(df)}")
            
        # Make sure model is trained
        if not hasattr(self.classifier, 'classes_'):
            print("Training classifier...")
            self.train_model()
            
        # Prepare texts
        texts = filtered_df['combined_text'].apply(self.preprocess_text).tolist()
        
        # Predict
        predictions, confidences = self.predict(texts)
        
        # Create a copy of the filtered DataFrame with confidences
        result_df = filtered_df.copy()
        result_df['confidence'] = confidences
        
        # Classify posts into categories using your original confidence threshold system
        def classify_confidence(confidence):
            if confidence >= confidence_threshold:
                return "high_confidence"
            elif confidence >= (confidence_threshold - borderline_margin):
                return "borderline"
            else:
                return "low_confidence"
        
        result_df['confidence_category'] = result_df['confidence'].apply(classify_confidence)
        
        # Enhanced company detection with NER
        result_df['mentioned_companies'] = result_df['combined_text'].apply(self.get_mentioned_companies)
        
        # Clean up
        result_df = result_df.drop(['has_mag7_term', 'combined_text'], axis=1)
        
        # Filter to only include high confidence and borderline
        final_df = result_df[result_df['confidence_category'].isin(['high_confidence', 'borderline'])]
        
        return final_df

# Example usage
if __name__ == "__main__":
    # Load your dataset
    reddit_data = pd.read_csv('reddit_posts_comments_additional.csv')  # Replace with your actual data file
    
    # Create classifier and filter posts
    classifier = BERTMag7Classifier(use_gpu=True)
    mag7_posts = classifier.filter_mag7_posts(reddit_data, confidence_threshold=0.6, borderline_margin=0.15)
    
    # Process different categories
    high_confidence_posts = mag7_posts[mag7_posts['confidence_category'] == 'high_confidence']
    borderline_posts = mag7_posts[mag7_posts['confidence_category'] == 'borderline']
    
    # Save to separate CSV files
    high_confidence_posts.to_csv('high_confidence_posts_additional.csv', index=False)
    borderline_posts.to_csv('borderline_posts_additional.csv', index=False)
    
    # Print summary
    print(f"Saved {len(high_confidence_posts)} high confidence posts to high_confidence_posts.csv")
    print(f"Saved {len(borderline_posts)} borderline posts to borderline_posts.csv")
    
    # Print results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(mag7_posts)