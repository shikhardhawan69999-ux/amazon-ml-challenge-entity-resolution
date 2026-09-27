import pandas as pd
import re
import unicodedata

NAME_ABBR = {
    "pvt": "private", "pvt.": "private", "priv": "private",
    "ltd": "limited", "ltd.": "limited", "lim": "limited",
    "co": "company", "co.": "company", "comp": "company",
    "corp": "corporation", "corp.": "corporation",
    "inc": "incorporated", "inc.": "incorporated",
    "llc": "limited liability company", "plc": "public limited company",
    "&": "and", "intl": "international", "int'l": "international",
    "ind": "industries", "inds": "industries", "tech": "technology",
    "technol": "technology", "enterp": "enterprise", "ent": "enterprise",
    "svc": "services", "svcs": "services", "grp": "group",
    "bros": "brothers", "assoc": "associates", "assn": "association"
}

LEGAL_SUFFIXES = {"private", "limited", "company", "corporation", "incorporated", "llc", "plc"}

ADDRESS_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "bldg": "building",
    "fl": "floor", "apt": "apartment", "ste": "suite",
    "hwy": "highway", "ln": "lane", "dr": "drive", "ct": "court",
    "pkwy": "parkway", "n": "north", "s": "south", "e": "east", "w": "west"
}

def normalize_unicode(text):
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in text if not unicodedata.combining(c))

def normalize_name(text):
    if pd.isna(text) or not text: return ""
    text = normalize_unicode(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = [NAME_ABBR.get(t, t) for t in text.split()]
    return " ".join(tokens)

def normalize_name_core(text):
    tokens = text.split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)

def normalize_address(text):
    if pd.isna(text) or not text: return ""
    text = normalize_unicode(text).lower()
    text = text.replace("&", " and ").replace("/", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = [ADDRESS_ABBR.get(t, t) for t in text.split()]
    return " ".join(tokens)

def extract_pincode(text):
    if pd.isna(text) or not text: return ""
    # Look for 5 or 6 digit numbers (US Zip or India PIN)
    match = re.search(r'\b\d{5,6}\b', str(text))
    return match.group(0) if match else ""

def preprocess_dataframe(df):
    print("Running advanced multi-view normalization...")
    if 'business_name' in df.columns:
        df['business_name_clean'] = df['business_name'].apply(normalize_name)
        df['business_name_core'] = df['business_name_clean'].apply(normalize_name_core)
        
    if 'business_address' in df.columns:
        df['business_address_clean'] = df['business_address'].apply(normalize_address)
        df['pincode'] = df['business_address'].apply(extract_pincode)
            
    df['combined_text'] = df['business_name_clean'] + " " + df['business_address_clean']
    
    # CRITICAL RAM FIX: Drop the massive raw string columns now that we have clean versions
    # This instantly frees up 3-4 GB of RAM!
    cols_to_drop = ['business_name', 'business_address', 'city', 'state', 'zip_code', 'business_description']
    for col in cols_to_drop:
        if col in df.columns:
            del df[col]
            
    import gc
    gc.collect()
    
    return df
