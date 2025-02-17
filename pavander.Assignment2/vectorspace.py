# Paul Vander Woude (pavander)

# Standard library imports
import sys
import math
from pathlib import Path

# Third-party imports
import numpy as np
from stopwords import get_stopwords

# Local application/library specific imports
from preprocess import tokenizeText
from porter_stemmer import PorterStemmer

# Global variables
STOP_WORDS = set(get_stopwords("en"))
STEMMER = PorterStemmer()

def preprocess(text, doc_id = None, verbose = 0):
    """Clean, tokenize, filter, and stem raw text."""
    tokens_unfiltered = tokenizeText(text)
    if (verbose >= 2):
        print(f'tokenized {doc_id}')

    tokens_unstemmed = [word for word in tokens_unfiltered if word not in STOP_WORDS]
    if (verbose >= 2):
        print(f'removed stop words from {doc_id}')

    tokens = [STEMMER.stem(word) for word in tokens_unstemmed]
    if (verbose >= 2):
        print(f'stemmed {doc_id}')
    
    return tokens


def add_tf_idf(tokens, inv_idx, doc_id, verbose = 0):
    for token in tokens:
        if token in inv_idx:
            inv_idx[token][doc_id] += 1
        else:
            inv_idx[token] = {doc_id: 1}

    return inv_idx


def indexDocument(doc_id, dw_mode, qw_mode, inv_idx, verbose = 0):
    """Add a document to the inverted index."""
    # ==================================================
    # STEP ONE: open file
    doc_path = Path(doc_id)
    if not doc_path.is_file():
        raise(FileNotFoundError(f'could not find file {doc_id}'))
    
    with doc_path.open("r", encoding="ISO-8859-1") as doc:
        if (verbose >= 2):
            print(f'opened {doc_id}')
    # ==================================================
    # STEP TWO: preprocess (clean, tokenize, and stem)
        doc_text_raw = doc.read()

        tokens = preprocess(doc_text_raw, doc_id, verbose)
        
    # ==================================================
    # STEP THREE: add tokens to inverted index
    if (dw_mode == 'tf.idf'):
        inv_idx = add_tf_idf(tokens, inv_idx, doc_id, verbose)
    else:
        raise(ValueError(f'unimplemented mode(s): d: {dw_mode} q: {qw_mode}'))

    return inv_idx


def construct_tf_idf(inv_idx, N, cos_norm=True, verbose=0):
    tf_idf = {}
    idf = {}
    doc_norms = {}

    for term, docs in inv_idx.items():  # Iterate correctly over dictionary
        df_t = len(docs)
        idf_t = np.log10(N / df_t)  # Compute IDF
        idf[term] = idf_t
        
        if verbose >= 1:
            print(f'Processed {term}: IDF = {idf_t}')

        tf_idf[term] = {}
        
        for doc, tf_t_d in docs.items():  # Iterate correctly over document dictionary
            w_t_d = tf_t_d * idf_t  # Compute TF-IDF weight
            tf_idf[term][doc] = w_t_d
            
            if cos_norm:
                if doc not in doc_norms:
                    doc_norms[doc] = 0
                doc_norms[doc] += w_t_d ** 2  # Sum squared weights for normalization
            
            if verbose >= 2:
                print(f'Processed {term} in doc {doc}: TF-IDF = {w_t_d}')

    # Apply cosine normalization if enabled
    if cos_norm:
        for doc in doc_norms:
            doc_norms[doc] = math.sqrt(doc_norms[doc])

    return tf_idf, idf, doc_norms


def retrieveDocuments(query, inv_idx, idf, doc_norms, dw_mode, qw_mode, cos_norm=True, verbose=0):
    """Retrieve relevant documents for a given query using cosine similarity."""
    tokens = preprocess(query, query, verbose)  # Tokenize & preprocess query
    results = {}

    if dw_mode == 'tf.idf' and qw_mode == 'tf.idf':  # Fixed '&' to 'and'
        q_vec = {}  # Query vector
        similarity_scores = {}

        # Compute query TF-IDF vector
        for token in tokens:
            tf = 1  # Assume binary term frequency (each token appears once in query)
            idf_t = idf.get(token, 0)  # Get IDF (default to 0 if token is unseen)
            q_weight = tf * idf_t  # Compute TF-IDF for query term
            q_vec[token] = q_weight

            # Accumulate similarity scores for documents containing the token
            if token in inv_idx:
                for doc, d_weight in inv_idx[token].items():
                    if doc not in similarity_scores:
                        similarity_scores[doc] = 0
                    similarity_scores[doc] += q_weight * d_weight  # Dot product

        if cos_norm:
            # Compute query norm
            query_norm = math.sqrt(sum(val**2 for val in q_vec.values()))

            # Normalize similarity scores
            for doc_id in similarity_scores:
                if doc_norms[doc_id] > 0 and query_norm > 0:  # Avoid division by zero
                    similarity_scores[doc_id] /= (query_norm * doc_norms[doc_id])

        # Sort results by similarity score
        results = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    return results



def main():
    """main/driver function."""

    # ==================================================
    # STEP ONE: PARSE CLI

    # check for sufficient arguments
    if (len(sys.argv) < 6):
        print('USAGE: vectorspace.py DOC_WEIGHT_MODE QUERY_WEIGHT_MODE COSINE_NORM(T/F) DOCUMENT_DIR TEST_QUERY_FILE (VERBOSE(0/1/2))')
        raise ValueError(f"Expected at least {5} arguments, got {len(sys.argv)}.")

    # argument 6: verbose flag
    verbose = 0
    if (len(sys.argv) == 7):
        verbose = max(sys.argv[6],2)

    # check for extra arguments
    if (len(sys.argv) > 7):
        print('USAGE: vectorspace.py DOC_WEIGHT_MODE QUERY_WEIGHT_MODE COSINE_NORM(T/F) DOCUMENT_DIR TEST_QUERY_FILE (VERBOSE(0/1/2))')
        raise ValueError(f"Expected at most {6} arguments, got {len(sys.argv)}.")
    
    # argument 1: weighting scheme for documents
    dw_mode = sys.argv[1]
    doc_weights = None
    if (dw_mode == 'tf.idf'):
        # tf.idf weighting scheme
        if (verbose):
            print('tf.idf document weighting scheme selected.')
        doc_weights = dict()
    else:
        # custom weighting scheme
        if (verbose):
            print('Custom document weighting scheme selected.')
        doc_weights = list()

    # argument 2: weighting scheme for documents
    qw_mode = sys.argv[2]
    query_weights = None
    if (qw_mode == 'tf.idf'):
        # tf.idf weighting scheme
        if (verbose):
            print('tf.idf query weighting scheme selected.')
        query_weights = dict()
    else:
        # custom weighting scheme
        if (verbose):
            print('Custom query weighting scheme selected.')
        query_weights = list()

    # argument 3: cosine normalizaiton
    cos_norm = None
    if (sys.argv[3] == 'T' | sys.argv[3] == 'True'):
        if (verbose):
            print('cosine normalization enabled')
        cos_norm = True
    elif (sys.argv[3] == 'F' | sys.argv[3] == 'False'):
        if (verbose):
            print('cosine normalization disabled')
        cos_norm = False
    else:
        raise(ValueError(f'unknown COSINE_NORM option: {sys.argv[3]} [expected: T/True OR F/False]'))
    
    # argument 4: document directory
    doc_dir_path = Path(sys.argv[4])
    if not doc_dir_path.is_dir():
        FileNotFoundError(f'The directory {sys.argv[4]} does not exist.')
    elif (verbose):
        print('document directory found successfully.')

    # argument 5: test query file
    test_query_filepath = Path(sys.argv[5])
    if not test_query_filepath.is_file():
        FileNotFoundError(f'The file {sys.argv[5]} does not exist.')
    elif (verbose):
        print('test query file found successfully.')

    # ==================================================
    # STEP TWO: TBA


if __name__ == "__main__":
    main()