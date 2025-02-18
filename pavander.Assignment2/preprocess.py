# Paul Vander Woude (pavander)

# Preprocessing & tokenization functions

import re
from contractions import fix
from stopwords import get_stopwords
from porter_stemmer import PorterStemmer

# Global variables
STOP_WORDS = set(get_stopwords("en"))
STEMMER = PorterStemmer()


def removeSGML(text):
    """Cleans raw text by removing SGML tags."""
    return re.sub(r"<[^>]+>", "", text)


def tokenizeText(text):
    """Tokenizes text, including removing SGML text and expanding contractions."""
    text = removeSGML(text)
    text = text.casefold()
    text = fix(text)
    
    # Regular expression for tokenizing
    pattern = r"""
        \b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b        # Dates (e.g., 01/31/2024, 2024-01-31)
        |\b(?:[A-Za-z]+\.){2,}                     # Acronyms (e.g., U.S.A., E.U.)
        |\b\w+(?:-\w+)+\b                          # Hyphenated words (e.g., mother-in-law)
        |\b\w+\b                                   # Words
        |\d+(?:,\d{3})*(?:\.\d+)?\b                # Numbers with commas/decimals (e.g., 1,000.50)
    """

    # Split the text into lines
    lines = text.splitlines()
    
    all_tokens = []
    
    for line in lines:
        # Tokenize each line, prepending <start> token
        tokens = re.findall(pattern, line, re.VERBOSE)
        
        # Process contractions and possessives
        final_tokens = []
        for token in tokens:
            if token.endswith("'s"):
                final_tokens.append(token[:-2])  # Remove 's'
                final_tokens.append("'s")        # Keep possessive separately
            else:
                final_tokens.append(token)

        # Add processed tokens to the final list
        all_tokens.extend(final_tokens)

    return all_tokens


def preprocess_and_tokenize(text, doc_id, verbose=0):
    """Clean, tokenize, filter, and stem raw text."""
    if verbose >= 2:
        print(f"[preprocess][L2] Starting preprocessing for '{doc_id}'.")
    
    tokens_unfiltered = tokenizeText(text)
    if verbose >= 3:
        print(f"[preprocess][L3] {doc_id}: Tokenized into {len(tokens_unfiltered)} tokens.")
    if verbose >= 4:
        print(f"[preprocess][L4] {doc_id}: Unfiltered tokens: {tokens_unfiltered}")

    tokens_unstemmed = [word for word in tokens_unfiltered if word not in STOP_WORDS]
    if verbose >= 3:
        print(f"[preprocess][L3] {doc_id}: Removed stop words; {len(tokens_unstemmed)} tokens remain.")
    if verbose >= 4:
        print(f"[preprocess][L4] {doc_id}: Tokens after stop word removal: {tokens_unstemmed}")

    tokens = [STEMMER.stem(word, 0, max(len(word) - 1, 0)) for word in tokens_unstemmed]
    if verbose >= 3:
        print(f"[preprocess][L3] {doc_id}: Stemming complete.")
    if verbose >= 4:
        print(f"[preprocess][L4] {doc_id}: Stemmed tokens: {tokens}")
    
    return tokens


def token_frequencies(tokens):
    """Convert a list of tokens into a dictionary of unique tokens and their frequencies."""
    freq_dict = {}
    for token in tokens:
        freq_dict[token] = freq_dict.get(token, 0) + 1
    return freq_dict


def extract_docid(filename):
    """Extracts the first number found in the filename as an integer."""
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"invalid filename {filename}.")