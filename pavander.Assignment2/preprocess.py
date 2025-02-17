# Paul Vander Woude (pavander) EECS 486 HW1 preprocess.py
import re
from contractions import fix

# PART 1: removeSGML

def removeSGML(text):
    """Cleans raw text by removing SGML tags."""
    return re.sub(r"<[^>]+>", "", text)


# PART 2: tokenizeText

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