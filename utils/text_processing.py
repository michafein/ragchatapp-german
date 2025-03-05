"""
Text processing utilities for the RAG chatbot.
Handles text formatting, cleaning, and sentence extraction.
"""

import re
import logging
from typing import List, Optional
import spacy
from config import Config

# Initialize logging
logger = logging.getLogger(__name__)

# SpaCy für Deutsch konfigurieren
try:
    # Sprachspezifischen Import basierend auf Konfiguration
    if hasattr(Config, 'LANGUAGE') and Config.LANGUAGE == "de":
        try:
            nlp = spacy.load("de_core_news_sm")
            logger.info("Deutsches SpaCy-Modell geladen")
        except OSError:
            logger.warning("Deutsches Sprachmodell nicht gefunden, installiere es mit: python -m spacy download de_core_news_sm")
            nlp = spacy.blank("de")
            nlp.add_pipe("sentencizer")
    else:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        
except Exception as e:
    logger.error(f"SpaCy-Initialisierung fehlgeschlagen: {e}")
    # Fallback-Satz-Splitting mit angepasster deutscher Variante
    def simple_sentence_split(text):
        # Deutsches Satz-Splitting mit Berücksichtigung von Anführungszeichen
        return re.split(r'(?<=[.!?»"\'])(?:\s+|$)', text)
    nlp = None

def split_german_compounds(text):
    """Teilt lange deutsche Komposita in ihre Bestandteile."""
    # Nur Wörter ohne Leerzeichen verarbeiten
    if ' ' in text or len(text) < 10:
        return [text]
    
    # Liste möglicher Trennstellen bei Komposita
    connectors = ["s", "es", "n", "en", "er", ""]
    min_part_length = 3  # Mindestlänge eines Wortteils
    
    parts = []
    for i in range(min_part_length, len(text) - min_part_length):
        for conn in connectors:
            if i + len(conn) >= len(text):
                continue
                
            if text[i:i+len(conn)] == conn:
                part1 = text[:i]
                part2 = text[i+len(conn):]
                
                # Prüfe, ob beide Teile lang genug sind
                if len(part1) >= min_part_length and len(part2) >= min_part_length:
                    parts.append(part1)
                    parts.append(part2)
    
    # Wenn keine Teile gefunden wurden, gib das Originalwort zurück
    return [text] if not parts else list(set([text] + parts))

def normalize_german_text(text):
    """Normalisiert deutschen Text für bessere Suche."""
    # Umlaute in alternative Schreibweise umwandeln
    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue',
        'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue',
        'ß': 'ss'
    }
    normalized_text = text
    for char, replacement in replacements.items():
        normalized_text = normalized_text.replace(char, replacement)
    
    return normalized_text

def get_normalized_variants(text):
    """Erzeugt normalisierte Varianten eines Textes für die Suche."""
    original = text.strip()
    normalized = normalize_german_text(original)
    
    # Wenn der Text keine Umlaute enthält, ist keine weitere Verarbeitung nötig
    if original == normalized:
        return [original]
    
    return [original, normalized]

def text_formatter(text: str) -> str:
    """
    Formats the text by removing line breaks and extra whitespace.
    
    Args:
        text: Raw input text
        
    Returns:
        Formatted text string
    """
    if not isinstance(text, str):
        logger.warning(f"Expected string but got {type(text)}")
        text = str(text)
    
    return text.replace("\n", " ").strip()

def clean_text_chunk(text: str) -> str:
    """
    Cleans text chunks by removing or escaping special characters.
    
    Args:
        text: Raw text chunk
        
    Returns:
        Cleaned text chunk
    """
    if not isinstance(text, str):
        logger.warning(f"Expected string but got {type(text)}")
        text = str(text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    
    # Properly escape special characters - fixed to preserve all backslashes
    text = text.replace('\\', '\\\\')  # First double all backslashes
    text = text.replace('"', '\\"')    # Then escape quotes
    
    # Replace newlines with spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def split_list(input_list: List[str], slice_size: int = 10) -> List[List[str]]:
    """
    Splits a list of sentences into smaller chunks of specified size.
    
    Args:
        input_list: List of strings to split
        slice_size: Maximum size of each chunk
        
    Returns:
        List of lists, where each inner list is a chunk of the original list
    """
    if not input_list:
        return []
    
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def extract_sentences(text: str) -> List[str]:
    """
    Extracts individual sentences from a text using spaCy.
    
    Args:
        text: Input text to process
        
    Returns:
        List of extracted sentences
    """
    if not text or not isinstance(text, str):
        return []
    
    # Use spaCy if available, otherwise use simple fallback
    if nlp:
        doc = nlp(text)
        return [str(sentence).strip() for sentence in doc.sents if str(sentence).strip()]
    else:
        # Fallback to simple sentence splitter
        return [s.strip() for s in simple_sentence_split(text) if s.strip()]

def ensure_complete_sentences(text: str) -> str:
    """
    Ensures text ends with a complete sentence by truncating at the last sentence-ending punctuation.
    
    Args:
        text: Input text
        
    Returns:
        Text ending with complete sentence
    """
    if not text:
        return ""
    
    # Find the position of the last sentence-ending punctuation
    last_punctuation = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    
    if last_punctuation != -1:
        # Return the text up to and including the punctuation
        return text[:last_punctuation + 1]
    else:
        # If no punctuation found, return the original text
        return text