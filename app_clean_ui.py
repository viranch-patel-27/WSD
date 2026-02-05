"""
Clean WSD UI with Dark/Light Mode Toggle
=========================================
- White/Dark background with toggle
- All text visible and readable
- Word-only selection (no punctuation)
"""

import os
import re
import torch
import pandas as pd
import streamlit as st
from transformers import BertTokenizerFast, BertForSequenceClassification

import nltk
from nltk.corpus import wordnet as wn

# Wikipedia integration for named entities
try:
    import wikipedia_knowledge
    import importlib
    importlib.reload(wikipedia_knowledge) # FORCE RELOAD to fix hanging issue
    from wikipedia_knowledge import get_wikipedia_summary, get_disambiguation_candidates
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

# Ensure resources
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Page config
st.set_page_config(
    page_title="Word Sense Disambiguation",
    page_icon="🎯",
    layout="centered"
)

# ---------------- CONFIG ----------------
MODEL_DIR = "bert-base-uncased"  # HuggingFace model for cloud deployment
MAX_LEN = 128
TOP_K = 6
ALPHA = 0.6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ---------------- THEME STATE ----------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# ---------------- APPLY THEME CSS ----------------
if st.session_state.dark_mode:
    # DARK MODE CSS
    st.markdown("""
    <style>
    /* Dark theme */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], 
    [data-testid="stToolbar"], [data-testid="stDecoration"], 
    [data-testid="stStatusWidget"], .main, .block-container {
        background-color: #1a1a2e !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ALL TEXT WHITE */
    *, p, span, div, label, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-size: 1.1rem;
    }

    .main-title {
        font-size: 5rem !important;
        font-weight: 800 !important;
        color: #60a5fa !important;
        margin-bottom: 0.5rem !important;
    }

    .subtitle {
        font-size: 1.2rem !important;
        color: #9ca3af !important;
        margin-bottom: 1rem;
    }

    /* Text area */
    .stTextArea textarea {
        background-color: #2d2d44 !important;
        border: 2px solid #4b5563 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-size: 1.2rem !important;
        caret-color: #ffffff !important;
        padding: 12px !important;
    }

    .stTextArea textarea:focus {
        border-color: #60a5fa !important;
    }

    /* Select box */
    .stSelectbox > div > div {
        background-color: #2d2d44 !important;
        border: 2px solid #4b5563 !important;
        border-radius: 8px !important;
    }

    .stSelectbox > div > div > div {
        color: #ffffff !important;
        font-size: 1.2rem !important;
    }

    /* Dropdown */
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
        background-color: #2d2d44 !important;
    }

    [data-baseweb="menu"] li, [role="option"] {
        background-color: #2d2d44 !important;
        color: #ffffff !important;
    }

    [data-baseweb="menu"] li:hover, [role="option"]:hover {
        background-color: #3d3d5c !important;
    }

    div[data-baseweb="select"] span {
        color: #ffffff !important;
    }

    /* Highlighted word */
    .highlight-word {
        background-color: #3b82f6;
        color: #ffffff !important;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.3rem !important;
        display: inline-block;
        margin: 4px;
    }

    .normal-word {
        color: #e5e7eb !important;
        display: inline-block;
        margin: 4px;
        padding: 6px 6px;
        font-size: 1.3rem !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #3b82f6 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        width: 100% !important;
        white-space: nowrap !important;
    }

    .stButton > button, .stButton > button *, .stButton > button span {
        color: #ffffff !important;
    }

    .stButton > button:hover {
        background-color: #2563eb !important;
    }

    /* Result box */
    .result-box {
        background-color: #1e3a5f;
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
    }

    .result-title {
        color: #60a5fa !important;
        font-size: 1.5rem !important;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .result-def {
        color: #e5e7eb !important;
        font-size: 1.2rem !important;
        line-height: 1.6;
    }

    /* Candidate item */
    .candidate-item {
        background-color: #2d2d44;
        border: 2px solid #4b5563;
        border-radius: 10px;
        padding: 16px;
        margin: 10px 0;
    }

    .candidate-item:hover {
        border-color: #3b82f6;
        background-color: #3d3d5c;
    }

    .candidate-name {
        color: #60a5fa !important;
        font-weight: 700;
        font-size: 1.2rem !important;
    }

    .candidate-def {
        color: #d1d5db !important;
        font-size: 1.1rem !important;
        margin-top: 6px;
    }

    .score-badge {
        display: inline-block;
        background-color: #3d3d5c;
        padding: 5px 10px;
        border-radius: 6px;
        margin-right: 8px;
        font-size: 1rem !important;
        color: #e5e7eb !important;
        font-weight: 600;
    }

    .section-divider {
        border-top: 2px solid #4b5563;
        margin: 20px 0;
    }

    /* Theme toggle button */
    .theme-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 999;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    # LIGHT MODE CSS
    st.markdown("""
    <style>
    /* Light theme */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], 
    [data-testid="stToolbar"], [data-testid="stDecoration"], 
    [data-testid="stStatusWidget"], .main, .block-container {
        background-color: #ffffff !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ALL TEXT BLACK */
    *, p, span, div, label, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-size: 1.1rem;
    }

    .main-title {
        font-size: 5rem !important;
        font-weight: 800 !important;
        color: #1e40af !important;
        margin-bottom: 0.5rem !important;
    }

    .subtitle {
        font-size: 1.2rem !important;
        color: #374151 !important;
        margin-bottom: 1rem;
    }

    /* Text area */
    .stTextArea textarea {
        background-color: #ffffff !important;
        border: 2px solid #d1d5db !important;
        border-radius: 8px !important;
        color: #000000 !important;
        font-size: 1.2rem !important;
        caret-color: #000000 !important;
        padding: 12px !important;
    }

    .stTextArea textarea:focus {
        border-color: #2563eb !important;
    }

    /* Select box */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 2px solid #d1d5db !important;
        border-radius: 8px !important;
    }

    .stSelectbox > div > div > div {
        color: #000000 !important;
        font-size: 1.2rem !important;
    }

    /* Dropdown */
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
        background-color: #ffffff !important;
    }

    [data-baseweb="menu"] li, [role="option"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    [data-baseweb="menu"] li:hover, [role="option"]:hover {
        background-color: #e5e7eb !important;
    }

    div[data-baseweb="select"] span {
        color: #000000 !important;
    }

    /* Highlighted word */
    .highlight-word {
        background-color: #2563eb;
        color: #ffffff !important;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.3rem !important;
        display: inline-block;
        margin: 4px;
    }

    .normal-word {
        color: #000000 !important;
        display: inline-block;
        margin: 4px;
        padding: 6px 6px;
        font-size: 1.3rem !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2563eb !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        width: 100% !important;
        white-space: nowrap !important;
    }

    .stButton > button, .stButton > button *, .stButton > button span {
        color: #ffffff !important;
    }

    .stButton > button:hover {
        background-color: #1d4ed8 !important;
    }

    /* Result box */
    .result-box {
        background-color: #dbeafe;
        border: 2px solid #93c5fd;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
    }

    .result-title {
        color: #1e40af !important;
        font-size: 1.5rem !important;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .result-def {
        color: #000000 !important;
        font-size: 1.2rem !important;
        line-height: 1.6;
    }

    /* Candidate item */
    .candidate-item {
        background-color: #f9fafb;
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 16px;
        margin: 10px 0;
    }

    .candidate-item:hover {
        border-color: #2563eb;
        background-color: #eff6ff;
    }

    .candidate-name {
        color: #1e40af !important;
        font-weight: 700;
        font-size: 1.2rem !important;
    }

    .candidate-def {
        color: #000000 !important;
        font-size: 1.1rem !important;
        margin-top: 6px;
    }

    .score-badge {
        display: inline-block;
        background-color: #e5e7eb;
        padding: 5px 10px;
        border-radius: 6px;
        margin-right: 8px;
        font-size: 1rem !important;
        color: #000000 !important;
        font-weight: 600;
    }

    .section-divider {
        border-top: 2px solid #e5e7eb;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- UTILS ----------------
def simple_tokenize(text):
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return text.split()

def extract_words_only(sentence):
    words = []
    tokens = sentence.split()
    for i, token in enumerate(tokens):
        clean_word = re.sub(r'[^\w]', '', token)
        if clean_word:
            words.append((i, clean_word, token))
    return words

def knowledge_score(context_tokens, synset):
    gloss = synset.definition()
    examples = " ".join(synset.examples())
    text = gloss + " " + examples
    for hyp in synset.hypernyms()[:2]:
        text += " " + hyp.definition()
    gloss_tokens = simple_tokenize(text)
    if not gloss_tokens:
        return 0.0
    return len(set(context_tokens) & set(gloss_tokens))

def is_likely_named_entity(word, sentence):
    """Check if word is likely a named entity (capitalized, not at start)."""
    # Check if word is capitalized in the original sentence
    tokens = sentence.split()
    for i, token in enumerate(tokens):
        clean = re.sub(r'[^\w]', '', token)
        if clean.lower() == word.lower():
            # Check capitalization (not just first word)
            if token[0].isupper() and (i > 0 or len(tokens) > 1):
                return True
    return False

def get_wikipedia_context(word, sentence):
    """
    Get Wikipedia context for any word, with compound term detection.
    Detects phrases like 'blood bank', 'apple tree', etc.
    """
    if not WIKIPEDIA_AVAILABLE:
        return None
    
    try:
        # Try to find compound terms (e.g., "blood bank", "apple tree")
        compound_term = find_compound_term(word, sentence)
        search_term = compound_term if compound_term else word
        
        # Pass sentence context for better disambiguation
        summary = get_wikipedia_summary(search_term, context=sentence)
        if summary:
            return {
                "summary": summary,
                "search_term": search_term,
                "is_compound": compound_term is not None,
                "word": word
            }
    except Exception as e:
        pass
    return None


def find_compound_term(word, sentence):
    """
    Find if the word is part of a compound term in the sentence.
    E.g., 'bank' in 'blood bank operates...' -> 'blood bank'
    Only returns compounds from the known list to avoid false positives.
    """
    # Common compound patterns to look for
    words = sentence.lower().split()
    word_lower = word.lower()
    
    # Find position of target word
    try:
        idx = words.index(word_lower)
    except ValueError:
        # Try partial match
        idx = -1
        for i, w in enumerate(words):
            if word_lower in w:
                idx = i
                break
        if idx == -1:
            return None
    
    # Words that should NOT form compound terms (auxiliary verbs, articles, etc.)
    skip_words = {
        'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
        'do', 'does', 'did', 'has', 'have', 'had', 'is', 'are', 'was', 'were',
        'the', 'a', 'an', 'to', 'and', 'or', 'but', 'for', 'with', 'at', 'by',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her',
        'this', 'that', 'these', 'those', 'some', 'any', 'all', 'each', 'every'
    }
    
    # Check for compound term (word before + target word)
    if idx > 0:
        prev_word = re.sub(r'[^a-z]', '', words[idx - 1])
        
        # Skip if previous word is an auxiliary or common function word
        if prev_word in skip_words:
            return None
            
        compound = f"{prev_word} {word_lower}"
        
        # List of known compound types - ONLY return compounds from this list
        known_compounds = [
            'blood bank', 'food bank', 'river bank', 'memory bank',
            'apple tree', 'apple pie', 'apple juice',
            'cell phone', 'prison cell', 'blood cell',
            'light bulb', 'traffic light', 'flash light',
            'book store', 'book shelf', 'comic book',
            'smart watch', 'pocket watch', 'stop watch',
            'wrist watch', 'night watch'
        ]
        
        if compound in known_compounds:
            return compound
    
    return None

def hybrid_predict(sentence, target_word):
    ctx_tokens = simple_tokenize(sentence)
    candidates = wn.synsets(target_word.lower())
    
    if not candidates:
        return None, []

    kb_scores = [knowledge_score(ctx_tokens, s) for s in candidates]
    max_kb = max(kb_scores) if kb_scores else 0

    sorted_idx = sorted(range(len(candidates)), key=lambda i: kb_scores[i], reverse=True)
    top_idx = sorted_idx[:TOP_K]
    norm_kb = [(kb_scores[i] / (max_kb + 1e-6)) for i in top_idx]

    bert_scores = []
    with torch.no_grad():
        for idx in top_idx:
            gloss = candidates[idx].definition()
            enc = tokenizer(sentence, gloss, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
            enc = {k: v.to(device) for k,v in enc.items()}
            out = model(**enc)
            prob = torch.sigmoid(out.logits).item()
            bert_scores.append(prob)

    combined = []
    for kb, nn, i_idx in zip(norm_kb, bert_scores, top_idx):
        hybrid = ALPHA * nn + (1 - ALPHA) * kb
        combined.append((hybrid, candidates[i_idx], kb, nn))

    combined.sort(key=lambda x: x[0], reverse=True)
    best = combined[0][1]
    return best, combined

# ---------------- UI LAYOUT ----------------

# Header with toggle button on same line
col1, col2 = st.columns([5, 2])
with col1:
    st.markdown('<h1 class="main-title">Word Sense Disambiguation</h1>', unsafe_allow_html=True)
with col2:
    st.markdown('<div style="height: 2.5rem;"></div>', unsafe_allow_html=True)  # Spacer for vertical alignment
    theme_label = "Light Mode" if st.session_state.dark_mode else "Dark Mode"
    st.button(theme_label, on_click=toggle_theme, key="theme_toggle")

st.markdown('<p class="subtitle">Enter a sentence and select a word to discover its meaning in context</p>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Input
st.markdown("### Enter your sentence:")
default_sentence = "I went to the bank to deposit money."
sentence = st.text_area("sentence_input", value=default_sentence, height=100, label_visibility="collapsed")

if sentence.strip():
    words_data = extract_words_only(sentence)
    
    if words_data:
        st.markdown("### Select a word to analyze:")
        word_options = [w[1] for w in words_data]
        
        selected_idx = st.selectbox(
            "word_select",
            range(len(words_data)),
            format_func=lambda i: word_options[i],
            label_visibility="collapsed"
        )
        
        target_word = words_data[selected_idx][1]
        target_original_idx = words_data[selected_idx][0]
        
        st.markdown("### Your sentence:")
        tokens = sentence.split()
        highlighted_html = ""
        for i, token in enumerate(tokens):
            if i == target_original_idx:
                highlighted_html += f'<span class="highlight-word">{token}</span> '
            else:
                highlighted_html += f'<span class="normal-word">{token}</span> '
        
        st.markdown(f'<div style="line-height: 2.5; margin: 16px 0;">{highlighted_html}</div>', unsafe_allow_html=True)
        
        st.markdown("")
        
        if st.button("Analyze Meaning"):
            with st.spinner("Analyzing..."):
                best, candidates = hybrid_predict(sentence, target_word)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            if best is None:
                st.error("No word senses found for this word.")
            else:
                # ALWAYS fetch Wikipedia context (not just for named entities)
                wiki_context = get_wikipedia_context(target_word, sentence)
                
                # Show Wikipedia context if found
                if wiki_context:
                    st.markdown("### 🌐 Wikipedia Context")
                    
                    # Show what was searched
                    search_display = wiki_context.get('search_term', target_word)
                    is_compound = wiki_context.get('is_compound', False)
                    
                    if is_compound:
                        header_text = f"📖 Compound Term: {search_display}"
                    elif is_likely_named_entity(target_word, sentence):
                        header_text = f"🏢 Named Entity: {search_display}"
                    else:
                        header_text = f"📖 Wikipedia: {search_display}"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #10b98122, #10b98111); 
                                border: 2px solid #10b981; border-radius: 12px; 
                                padding: 20px; margin: 16px 0;">
                        <div style="font-size: 1.3rem; font-weight: 700; margin-bottom: 10px; color: #10b981;">
                            {header_text}
                        </div>
                        <div style="font-size: 1.1rem; line-height: 1.6;">
                            {wiki_context['summary'][:500]}{'...' if len(wiki_context['summary']) > 500 else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                else:
                    # Show message that Wikipedia couldn't find anything
                    st.markdown("### 🌐 Wikipedia Context")
                    st.info(f"No Wikipedia article found for '{target_word}'")
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                st.markdown("### 📚 WordNet Analysis")
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-title">{best.name()}</div>
                    <div class="result-def">{best.definition()}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if best.examples():
                    st.markdown("### Example usage:")
                    for ex in best.examples()[:2]:
                        st.markdown(f'> *"{ex}"*')
                
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                st.markdown("### All Possible Meanings")
                
                for i, (score, syn, kb, nn) in enumerate(candidates):
                    rank = i + 1
                    st.markdown(f"""
                    <div class="candidate-item">
                        <div class="candidate-name">#{rank} {syn.name()}</div>
                        <div class="candidate-def">{syn.definition()}</div>
                        <div class="candidate-scores">
                            <span class="score-badge">Score: {score:.1%}</span>
                            <span class="score-badge">Neural: {nn:.1%}</span>
                            <span class="score-badge">Knowledge: {kb:.1f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("No valid words found in the sentence.")
else:
    st.info("Enter a sentence above to get started!")

# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1rem;">Powered by BERT + WordNet + Wikipedia Hybrid Model</p>', unsafe_allow_html=True)
