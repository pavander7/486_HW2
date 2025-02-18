# Paul Vander Woude (pavander)

# Standard library imports
import sys
import math
import random
import re
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

    tokens = [STEMMER.stem(word, 0, max(len(word) - 1, 0)) for word in tokens_unstemmed]
    if verbose >= 3:
        print(f"[preprocess][L3] {doc_id}: Stemming complete.")
    if verbose >= 4:
        print(f"[preprocess][L4] {doc_id}: Stemmed tokens: {tokens}")
    
    return tokens


def load_reljudge(reljudge_filepath, verbose=0):
    """Load relevance judgments from a file."""
    reljudge = {}
    with open(reljudge_filepath, "r", encoding="ISO-8859-1") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            qid = int(parts[0])
            doc_id = parts[1]
            reljudge.setdefault(qid, []).append(doc_id)
    if verbose >= 1:
        print(f"[load_reljudge] Loaded relevance judgments for {len(reljudge)} queries.")
    return reljudge


def evaluate_query(query_id, retrieved_docs, reljudge, all_doc_ids, cutoffs=(10, 50, 100, 500)):
    """Compute precision and recall for a single query at various cutoffs."""
    results = []
    # Get the set of relevant document IDs for this query
    relevant = set(reljudge.get(query_id, []))
    total_rel = len(relevant)
    
    # For each cutoff, pad if necessary.
    for cutoff in cutoffs:
        # Copy the current ranking.
        current_ranking = list(retrieved_docs)
        if len(current_ranking) < cutoff:
            needed = cutoff - len(current_ranking)
            # Choose padding docs randomly from those not already in current_ranking.
            available = list(all_doc_ids - set(current_ranking))
            if len(available) < needed:
                # In the unlikely event that there are not enough docs to pad, use all available.
                pad_docs = available
            else:
                pad_docs = random.sample(available, needed)
            current_ranking.extend(pad_docs)
        else:
            current_ranking = current_ranking[:cutoff]
            
        # Evaluate: count how many of these docs are relevant.
        num_rel_ret = sum(1 for d in current_ranking if d in relevant)
        precision = num_rel_ret / cutoff
        recall = (num_rel_ret / total_rel) if total_rel > 0 else 0.0
        results.append((query_id, cutoff, precision, recall))
    return results


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


def extract_docid(filename):
    """Extracts the first number found in the filename as an integer."""
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"invalid filename {filename}.")


def indexDocument(doc_filepath, dw_mode, qw_mode, inv_idx, doc_lengths = None, verbose=0):
    """Add a document to the inverted index."""
    doc_path = Path(doc_filepath)
    doc_id = extract_docid(doc_path.name)
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
        
        if dw_mode in ('tf.idf', 'bm25'):
            add_tf_idf(tokens, inv_idx, doc_id, verbose)
            if dw_mode == 'bm25':
                doc_lengths[doc_id] = len(tokens)
                if verbose >= 3:
                    print(f"[indexDocument][L3] Recorded doc length {doc_lengths[doc_id]} for '{doc_id}'.")
            if verbose >= 3:
                print(f"[indexDocument][L3] Tokens added to the inverted index for '{doc_id}'.")
        else:
            raise ValueError(f"Unimplemented mode(s): d: {dw_mode} q: {qw_mode}")


def construct_weights(inv_idx, N, dw_mode, cos_norm=False, doc_lengths=None, k1=1.5, b=0.75, verbose=0):
    """Construct weighting scores from the inverted index."""
    weights = {}
    idf = {}
    doc_norms = {} if cos_norm else None

    if dw_mode == "bm25":
        if doc_lengths is None:
            raise ValueError("doc_lengths must be provided for BM25 weighting.")
        avgdl = np.mean(list(doc_lengths.values()))
        if verbose >= 2:
            print(f"[construct_weights][L2] Constructing BM25 weights with k1={k1}, b={b}, avgdl={avgdl:.2f}.")
    elif dw_mode == "tf.idf":
        if verbose >= 2:
            print("[construct_weights][L2] Constructing TF-IDF weights.")
    else:
        raise ValueError(f"Unknown weighting scheme {dw_mode}. Use 'tf.idf' or 'bm25'.")

    for term, docs in inv_idx.items():
        df_t = len(docs)
        idf_t = None
        if dw_mode == "tf.idf":
            idf_t = np.log10(N / df_t)
        elif dw_mode == "bm25":
            idf_t = np.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
        idf[term] = idf_t
        if verbose >= 3:
            print(f"[construct_weights][L3] Term '{term}': DF = {df_t}, IDF = {idf_t:.4f}")

        weights[term] = {}
        for doc, freq in docs.items():
            if dw_mode == "tf.idf":
                w = freq * idf_t
            elif dw_mode == "bm25":
                norm = 1 - b + b * (doc_lengths[doc] / avgdl)
                w = idf_t * ((freq * (k1 + 1)) / (freq + k1 * norm))
            weights[term][doc] = w
            if cos_norm:
                doc_norms[doc] = doc_norms.get(doc, 0) + w ** 2
            if verbose >= 4:
                print(f"[construct_weights][L4] Term '{term}' in doc '{doc}': freq = {freq}, weight = {w:.4f}")

    if cos_norm:
        for doc in doc_norms:
            doc_norms[doc] = math.sqrt(doc_norms[doc])
            if verbose >= 3:
                print(f"[construct_weights][L3] Document '{doc}': Norm = {doc_norms[doc]:.4f}")

    return weights, idf, (doc_norms if cos_norm else {})


def compute_query_tf_idf(query, idf, verbose=0):
    q_vec = {}
    for token in query:
        tf_val = 1  # Binary term frequency assumed.
        idf_t = idf.get(token, 0)
        q_weight = tf_val * idf_t
        q_vec[token] = q_weight
        if verbose >= 4:
            print(f"[compute_query_tf_idf][L4] Query token '{token}': TF = {tf_val}, IDF = {idf_t:.4f}, weight = {q_weight:.4f}")
    return q_vec


def compute_query_bm25(query, idf, verbose=0):
    q_vec = {}
    for token in query:
        idf_t = idf.get(token, 0)
        q_vec[token] = idf_t
        if verbose >= 4:
            print(f"[compute_query_bm25][L4] Query token '{token}': IDF = {idf_t:.4f}, weight = {idf_t:.4f}")
    return q_vec


def retrieveDocuments(query, index, idf, doc_norms, qw_mode, cos_norm=False, verbose=0):
    """Retrieve relevant documents for a given query using cosine similarity."""
    if verbose >= 2:
        print(f"[retrieveDocuments][L2] Retrieving documents for query: {query.strip()}")
    
    tokens = preprocess(query, 'query', verbose if verbose >= 4 else 0)
    if verbose >= 3:
        print(f"[retrieveDocuments][L3] Query tokenized into {len(tokens)} tokens: {tokens}")
    
    results = {}

    # Compute query vector based on the selected query weighting scheme
    if qw_mode == 'tf.idf':
        q_vec = compute_query_tf_idf(tokens, idf, verbose)
    elif qw_mode == 'bm25':
        q_vec = compute_query_bm25(tokens, idf, verbose)
    else:
        raise ValueError("Unimplemented query weighting mode")

    # Now compute similarity scores.
    scores = {}
    # Loop over tokens
    for token in tokens:
        # Get document weights based on the selected document weighting scheme
        if token in index:
            for doc, d_weight in index[token].items():
                score = scores.get(doc, 0) + q_vec.get(token, 0) * d_weight
                scores[doc] = score
                if verbose >= 4:
                    print(f"[retrieveDocuments][L4] Accumulated for doc '{doc}': token '{token}', token_score = {score:.4f}, cumulative score = {scores[doc]:.4f}")
    
    if cos_norm:
        # Compute query norm for query vector.
        query_norm = math.sqrt(sum(val**2 for val in q_vec.values()))
        if verbose >= 3:
            print(f"[retrieveDocuments][L3] Query norm = {query_norm:.4f}")
        for doc in scores:
            if doc_norms.get(doc, 0) > 0 and query_norm > 0:
                scores[doc] /= (query_norm * doc_norms[doc])
                if verbose >= 4:
                    print(f"[retrieveDocuments][L4] Doc '{doc}': Normalized score = {scores[doc]:.4f}")

    results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if verbose >= 3:
        print(f"[retrieveDocuments][L3] Retrieval complete. {len(results)} documents found.")

    return results


def main():
    """Main/driver function."""
    # STEP ONE: PARSE CLI
    # Check help flag
    if len(sys.argv) >= 2 and sys.argv[1] in ('H', 'h', '-h', '-H', 'help', 'Help', 'HELP', '--help', '--Help', '--HELP'):
        print("USAGE: vectorspace.py DOC_WEIGHT_MODE QUERY_WEIGHT_MODE COSINE_NORM DOCUMENT_DIR TEST_QUERY_FILE ANSWERS [RELJUDGE_FILE] (VERBOSE)")
        print(
            "\tDOC_WEIGHT MODE: tf.idf, bm25\n" +
            "\tQUERY_WEIGHT MODE: tf.idf, bm25\n" +
            "\tCOSINE_NORM: T/True/TRUE/true/1 OR F/False/FALSE/false/0\n" +
            "\tDOCUMENT_DIR: Directory containing documents\n" +
            "\tTEST_QUERY_FILE: File containing test queries\n" +
            "\tANSWERS: Enable answers evaluation (T/True/TRUE/true/1) or disable (F/False/FALSE/false/0)\n" +
            "\tRELJUDGE_FILE: (Required if ANSWERS enabled) File with relevance judgments\n" +
            "\tVERBOSE: (optional) Verbosity level (default 0)\n"
        )
        return 0
    
    # We need at least 7 arguments if answers is disabled, and 8 if enabled.
    # argv[0] is the script name.
    min_args_without_answers = 7  # indices 0..6: 6 arguments plus script name.
    min_args_with_answers = 8

    # Determine if answers evaluation is enabled.
    # We'll assume the argument order:
    # [1] dw_mode, [2] qw_mode, [3] cos_norm, [4] doc_dir, [5] test_query_file, [6] answers, [7] (reljudge_file if answers enabled) , [optional 8] verbose
    if len(sys.argv) < min_args_without_answers:
        print("USAGE: vectorspace.py DOC_WEIGHT_MODE QUERY_WEIGHT_MODE COSINE_NORM DOCUMENT_DIR TEST_QUERY_FILE ANSWERS [RELJUDGE_FILE] (VERBOSE)")
        raise ValueError(f"Expected at least 6 arguments, got {len(sys.argv)-1}.")
    
    # Take in args
    dw_mode = sys.argv[1]
    qw_mode = sys.argv[2]
    cos_norm_arg = sys.argv[3]
    doc_dir_str = sys.argv[4]
    test_query_filepath_str = sys.argv[5]
    answers_arg = sys.argv[6]

    # Determine verbose level.
    verbose = 0
    # If answers is enabled, then we expect an extra argument for reljudge_file, so the position of verbose changes.
    if answers_arg in ('T', 'True', 'TRUE', 'true', '1'):
        answers = True
        if len(sys.argv) < min_args_with_answers:
            raise ValueError("When answers is enabled, a reljudge_file must be provided.")
        reljudge_filepath_str = sys.argv[7]
        if len(sys.argv) >= 9:
            verbose = int(sys.argv[8])
    elif answers_arg in ('F', 'False', 'FALSE', 'false', '0'):
        answers = False
        if len(sys.argv) >= 7:
            # If answers disabled, then verbose might be at position 7.
            if len(sys.argv) >= 8:
                verbose = int(sys.argv[7])
    else:
        raise ValueError("Unknown ANSWERS option. Expected T/True/TRUE/true/1 or F/False/FALSE/false/0")

    # Parse cos_norm option.
    if cos_norm_arg in ('T', 'True', 'TRUE', 'true', '1'):
        cos_norm = True
        if verbose >= 1:
            print("[main][L1] Cosine normalization enabled.")
    elif cos_norm_arg in ('F', 'False', 'FALSE', 'false', '0'):
        cos_norm = False
        if verbose >= 1:
            print("[main][L1] Cosine normalization disabled.")
    else:
        raise ValueError(f"Unknown COSINE_NORM option: {cos_norm_arg}")
    
    # Validate dw_mode and qw_mode.
    if dw_mode == 'tf.idf':
        if verbose >= 1:
            print("[main][L1] tf.idf document weighting scheme selected.")
    elif dw_mode == 'bm25':
        if verbose >= 1:
            print("[main][L1] bm25 document weighting scheme selected.")
    else:
        raise ValueError(f"Unknown document weighting scheme {dw_mode} selected.")

    if qw_mode == 'tf.idf':
        if verbose >= 1:
            print("[main][L1] tf.idf query weighting scheme selected.")
    elif qw_mode == 'bm25':
        if verbose >= 1:
            print("[main][L1] bm25 query weighting scheme selected.")
    else:
        raise ValueError(f"Unknown query weighting scheme {qw_mode} selected.")

    # Validate document directory and test query file.
    doc_dir_path = Path(doc_dir_str)
    if not doc_dir_path.is_dir():
        raise FileNotFoundError(f"The directory {doc_dir_str} does not exist.")
    else:
        if verbose >= 1:
            print("[main][L1] Document directory found successfully.")

    test_query_filepath = Path(test_query_filepath_str)
    if not test_query_filepath.is_file():
        raise FileNotFoundError(f"The file {test_query_filepath_str} does not exist.")
    else:
        if verbose >= 1:
            print("[main][L1] Test query file found successfully.")

    # If answers is enabled, load the relevance judgments.
    if answers:
        reljudge_filepath = Path(reljudge_filepath_str)
        if not reljudge_filepath.is_file():
            raise FileNotFoundError(f"The reljudge file {reljudge_filepath_str} does not exist.")
        reljudge = load_reljudge(reljudge_filepath, verbose)
    else:
        reljudge = {}

    # STEP TWO: READ DOCUMENTS
    filepaths = list(doc_dir_path.iterdir())
    if verbose >= 1:
        print(f"[main][L1] Found {len(filepaths)} files in document directory.")
    # Create a set of all document IDs for padding later.
    if answers:
        all_doc_ids = set(file.name for file in filepaths)

    # STEP THREE: INDEX FILES
    inv_idx = {}
    doc_lengths = {}
    for file in filepaths:
        indexDocument(file, dw_mode, qw_mode, inv_idx, doc_lengths, verbose)
    if verbose >= 1:
        print(f"[main][L1] Indexing complete. Vocabulary size = {len(inv_idx)}")

    # STEP FOUR: CALCULATE WEIGHTS
    index, idf, doc_norms = construct_weights(inv_idx, len(filepaths), dw_mode, cos_norm, doc_lengths, verbose=verbose)
    if verbose >= 1:
        print(f"[main][L1] Scoring complete. Vocabulary size = {len(index)}")
    
    # Set output filename based on weighting modes and cos_norm.
    output_filepath = Path(f"cranfield.{dw_mode}.{qw_mode}.{int(cos_norm)}.output")

    # We'll also prepare a filename for answers if enabled.
    if answers:
        answers_filepath = output_filepath.with_suffix(".answers")
    
    # STEP FIVE: PROCESS QUERIES AND RETRIEVE DOCUMENTS
    # We'll store the evaluation results per query (if answers enabled).
    eval_results = []  # list of tuples (query_id, cutoff, precision, recall)
    with open(test_query_filepath, "r", encoding="ISO-8859-1") as query_file, \
         open(output_filepath, "w", encoding="ISO-8859-1") as output_file:
        for line in query_file:
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            query_id = int(parts[0])
            query = parts[1].strip()
            if verbose >= 2:
                print(f"[main][L2] Processing query {query_id}: {query}")
            results = retrieveDocuments(query, index, idf, doc_norms, qw_mode, cos_norm, verbose)
            for doc_id, score in results:
                output_file.write(f"{query_id} {doc_id} {score}\n")
            if verbose >= 2:
                print(f"[main][L2] Query {query_id}: Retrieved {len(results)} documents.")
            # If answers evaluation is enabled, evaluate this query.
            if answers:
                # Extract just the document IDs from the results (in order).
                retrieved_docs = [doc_id for doc_id, _ in results]
                # Evaluate using cutoffs 10, 50, 100, 500.
                eval_for_query = evaluate_query(query_id, retrieved_docs, reljudge, all_doc_ids)
                eval_results.extend(eval_for_query)

    if verbose >= 1:
        print(f"[main][L1] Output written to {output_filepath}")

    # STEP SIX: IF ANSWERS ENABLED, OUTPUT THE EVALUATION RESULTS
    if answers:
        with open(answers_filepath, "w", encoding="ISO-8859-1") as ans_file:
            # First line: configuration info.
            cos_norm_status = "enabled" if cos_norm else "disabled"
            ans_file.write(f"dw_mode: {dw_mode}, qw_mode: {qw_mode}, cos_norm: {cos_norm_status}\n")
            # Then one line per evaluation result.
            for query_id, cutoff, precision, recall in eval_results:
                ans_file.write(f"{query_id} {cutoff} {precision:.4f} {recall:.4f}\n")
        if verbose >= 1:
            print(f"[main][L1] Answers evaluation output written to {answers_filepath}")


if __name__ == "__main__":
    main()
