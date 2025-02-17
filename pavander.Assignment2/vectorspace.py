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


def preprocess(text, doc_id, verbose=0):
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

    # Note: assuming the stemmer's stem method takes parameters as shown.
    tokens = [STEMMER.stem(word, 0, max(len(word) - 1, 0)) for word in tokens_unstemmed]
    if verbose >= 3:
        print(f"[preprocess][L3] {doc_id}: Stemming complete.")
    if verbose >= 4:
        print(f"[preprocess][L4] {doc_id}: Stemmed tokens: {tokens}")
    
    return tokens


def add_tf_idf(tokens, inv_idx, doc_id, verbose=0):
    """Add tokens to the inverted index, updating term frequencies."""
    if verbose >= 2:
        print(f"[add_tf_idf][L2] Adding tokens for document '{doc_id}'.")
    for token in tokens:
        if token not in inv_idx:
            inv_idx[token] = {}
            if verbose >= 4:
                print(f"[add_tf_idf][L4] New token '{token}' added to index.")
        if doc_id not in inv_idx[token]:
            inv_idx[token][doc_id] = 0
            if verbose >= 4:
                print(f"[add_tf_idf][L4] Initializing count for token '{token}' in doc '{doc_id}'.")
        inv_idx[token][doc_id] += 1
        if verbose >= 4:
            print(f"[add_tf_idf][L4] Token '{token}' in doc '{doc_id}' count: {inv_idx[token][doc_id]}.")


def indexDocument(doc_filepath, dw_mode, qw_mode, inv_idx, verbose=0):
    """Add a document to the inverted index."""
    doc_path = Path(doc_filepath)
    doc_id = doc_path.name
    if not doc_path.is_file():
        raise FileNotFoundError(f"Could not find file {doc_id}")
    
    if verbose >= 2:
        print(f"[indexDocument][L2] Indexing document '{doc_id}'.")
    
    with doc_path.open("r", encoding="ISO-8859-1") as doc:
        if verbose >= 3:
            print(f"[indexDocument][L3] Opened file '{doc_id}'.")
        doc_text_raw = doc.read()
        if verbose >= 4:
            print(f"[indexDocument][L4] Read {len(doc_text_raw)} characters from '{doc_id}'.")
        tokens = preprocess(doc_text_raw, doc_id, verbose)
        if verbose >= 3:
            print(f"[indexDocument][L3] Preprocessing complete for '{doc_id}'; {len(tokens)} tokens obtained.")
        
        if dw_mode == 'tf.idf':
            add_tf_idf(tokens, inv_idx, doc_id, verbose)
            if verbose >= 3:
                print(f"[indexDocument][L3] Tokens added to the inverted index for '{doc_id}'.")
        else:
            raise ValueError(f"Unimplemented mode(s): d: {dw_mode} q: {qw_mode}")


def construct_tf_idf(inv_idx, N, cos_norm=False, verbose=0):
    """Construct TF-IDF weights from the inverted index."""
    if verbose >= 2:
        print("[construct_tf_idf][L2] Constructing TF-IDF weights.")
    
    tf_idf = {}
    idf = {}
    doc_norms = {}

    for term, docs in inv_idx.items():
        df_t = len(docs)
        idf_t = np.log10(N / df_t)
        idf[term] = idf_t
        if verbose >= 3:
            print(f"[construct_tf_idf][L3] Term '{term}': DF = {df_t}, IDF = {idf_t:.4f}")

        tf_idf[term] = {}
        for doc, tf_t_d in docs.items():
            w_t_d = tf_t_d * idf_t
            tf_idf[term][doc] = w_t_d
            if cos_norm:
                doc_norms[doc] = doc_norms.get(doc, 0) + w_t_d ** 2
            if verbose >= 4:
                print(f"[construct_tf_idf][L4] Term '{term}' in doc '{doc}': TF = {tf_t_d}, Weight = {w_t_d:.4f}")

    if cos_norm:
        for doc in doc_norms:
            doc_norms[doc] = math.sqrt(doc_norms[doc])
            if verbose >= 3:
                print(f"[construct_tf_idf][L3] Document '{doc}': Norm = {doc_norms[doc]:.4f}")

    return tf_idf, idf, doc_norms


def retrieveDocuments(query, inv_idx, idf, doc_norms, dw_mode, qw_mode, cos_norm=False, verbose=0):
    """Retrieve relevant documents for a given query using cosine similarity."""
    if verbose >= 2:
        print(f"[retrieveDocuments][L2] Retrieving documents for query: {query.strip()}")
    
    tokens = preprocess(query, 'query', verbose if verbose >= 4 else 0)
    if verbose >= 3:
        print(f"[retrieveDocuments][L3] Query tokenized into {len(tokens)} tokens: {tokens}")
    
    results = {}

    if dw_mode == 'tf.idf' and qw_mode == 'tf.idf':
        q_vec = {}
        similarity_scores = {}

        for token in tokens:
            tf_val = 1  # Binary term frequency assumed.
            idf_t = idf.get(token, 0)
            q_weight = tf_val * idf_t
            q_vec[token] = q_weight
            if verbose >= 4:
                print(f"[retrieveDocuments][L4] Query token '{token}': TF = {tf_val}, IDF = {idf_t:.4f}, weight = {q_weight:.4f}")
            if token in inv_idx:
                for doc, d_weight in inv_idx[token].items():
                    similarity_scores[doc] = similarity_scores.get(doc, 0) + q_weight * d_weight
                    if verbose >= 4:
                        print(f"[retrieveDocuments][L4] Accumulated for doc '{doc}': token '{token}', d_weight = {d_weight:.4f}, cumulative score = {similarity_scores[doc]:.4f}")

        if cos_norm:
            query_norm = math.sqrt(sum(val**2 for val in q_vec.values()))
            if verbose >= 3:
                print(f"[retrieveDocuments][L3] Query norm = {query_norm:.4f}")
            for doc_id in similarity_scores:
                if doc_norms.get(doc_id, 0) > 0 and query_norm > 0:
                    similarity_scores[doc_id] /= (query_norm * doc_norms[doc_id])
                    if verbose >= 4:
                        print(f"[retrieveDocuments][L4] Doc '{doc_id}': Normalized score = {similarity_scores[doc_id]:.4f}")

        results = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        if verbose >= 3:
            print(f"[retrieveDocuments][L3] Retrieval complete. {len(results)} documents found.")

    return results


def main():
    """Main/driver function."""
    # STEP ONE: PARSE CLI
    if len(sys.argv) < 6:
        print("USAGE: vectorspace.py DOC_WEIGHT_MODE QUERY_WEIGHT_MODE COSINE_NORM(T/F) DOCUMENT_DIR TEST_QUERY_FILE (VERBOSE(0/1/2/3/4))")
        raise ValueError(f"Expected at least 5 arguments, got {len(sys.argv)}.")

    # Determine verbose level (0 disables all verbose output)
    verbose = 0
    if len(sys.argv) == 7:
        verbose = int(sys.argv[6])
    if len(sys.argv) > 7:
        print("USAGE: vectorspace.py DOC_WEIGHT_MODE QUERY_WEIGHT_MODE COSINE_NORM(T/F) DOCUMENT_DIR TEST_QUERY_FILE (VERBOSE(0/1/2/3/4))")
        raise ValueError(f"Expected at most 6 arguments, got {len(sys.argv)}.")

    # Only print L1 messages if verbose >= 1
    if verbose >= 1:
        if sys.argv[1] == 'tf.idf':
            print("[main][L1] tf.idf document weighting scheme selected.")
        else:
            print("[main][L1] Custom document weighting scheme selected.")

        if sys.argv[2] == 'tf.idf':
            print("[main][L1] tf.idf query weighting scheme selected.")
        else:
            print("[main][L1] Custom query weighting scheme selected.")

    cos_norm = None
    if sys.argv[3] in ('T', 'True'):
        cos_norm = True
        if verbose >= 1:
            print("[main][L1] Cosine normalization enabled.")
    elif sys.argv[3] in ('F', 'False'):
        cos_norm = False
        if verbose >= 1:
            print("[main][L1] Cosine normalization disabled.")
    else:
        raise ValueError(f"Unknown COSINE_NORM option: {sys.argv[3]} [expected: T/True OR F/False]")
    
    doc_dir_path = Path(sys.argv[4])
    if not doc_dir_path.is_dir():
        raise FileNotFoundError(f"The directory {sys.argv[4]} does not exist.")
    else:
        if verbose >= 1:
            print("[main][L1] Document directory found successfully.")

    test_query_filepath = Path(sys.argv[5])
    if not test_query_filepath.is_file():
        raise FileNotFoundError(f"The file {sys.argv[5]} does not exist.")
    else:
        if verbose >= 1:
            print("[main][L1] Test query file found successfully.")

    # STEP TWO: READ DOCUMENTS
    filepaths = list(doc_dir_path.iterdir())
    if verbose >= 1:
        print(f"[main][L1] Found {len(filepaths)} files in document directory.")
    inv_idx = {}

    # STEP THREE: INDEX FILES
    for file in filepaths:
        indexDocument(file, sys.argv[1], sys.argv[2], inv_idx, verbose)
    if verbose >= 1:
        print(f"[main][L1] Indexing complete. Vocabulary size = {len(inv_idx)}")

    # STEP FOUR: CALCULATE WEIGHTS
    if sys.argv[1] == 'tf.idf':
        inv_idx, idf, doc_norms = construct_tf_idf(inv_idx, len(filepaths), cos_norm, verbose)
    if verbose >= 1:
        print(f"[main][L1] Scoring complete. Vocabulary size = {len(inv_idx)}")
    
    output_filepath = Path(f"cranfield.{sys.argv[1]}.{sys.argv[2]}.{int(cos_norm)}.output")

    # STEP FIVE: PROCESS QUERIES AND RETRIEVE DOCUMENTS
    with open(test_query_filepath, "r", encoding="ISO-8859-1") as query_file, \
         open(output_filepath, "w", encoding="ISO-8859-1") as output_file:
        for line in query_file:
            parts = line.split(maxsplit=1)
            query_id = int(parts[0])
            query = parts[1]
            if verbose >= 2:
                print(f"[main][L2] Processing query {query_id}: {query.strip()}")
            results = retrieveDocuments(query, inv_idx, idf, doc_norms, sys.argv[1], sys.argv[2], cos_norm, verbose)
            for doc_id, score in results:
                output_file.write(f"{query_id} {doc_id} {score}\n")
            if verbose >= 2:
                print(f"[main][L2] Query {query_id}: Retrieved {len(results)} documents.")

    if verbose >= 1:
        print(f"[main][L1] Output written to {output_filepath}")


if __name__ == "__main__":
    main()
