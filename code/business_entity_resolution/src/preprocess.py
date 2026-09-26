import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def expand_abbreviations(text):
    if not text:
        return text
    abbr_dict = {
        r'\bcorp\b': 'corporation',
        r'\bpvt\b': 'private',
        r'\bltd\b': 'limited',
        r'\bst\b': 'street',
        r'\brd\b': 'road',
        r'\binc\b': 'incorporated',
        r'\bco\b': 'company',
        r'\bllc\b': 'limited liability company',
        r'\bintl\b': 'international',
        r'\bmfg\b': 'manufacturing'
    }
    for abbr, full in abbr_dict.items():
        text = re.sub(abbr, full, text)
    return text

def preprocess_dataframe(df):
    for col in ['business_name', 'business_address']:
        if col in df.columns:
            df[col + '_clean'] = df[col].apply(clean_text).apply(expand_abbreviations)
            
    # Create combined text for robust TF-IDF blocking
    df['combined_text'] = df['business_name_clean'] + " " + df['business_address_clean']
    return df
