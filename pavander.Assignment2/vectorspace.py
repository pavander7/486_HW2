# Paul Vander Woude (pavander)

import sys
import tokenize
import pathlib

def indexDocument(doc_id, doc_weights, query_weights, inv_idx):
    return inv_idx


def retrieveDocuments(query, inv_idx, doc_weights, query_weights):
    results = dict()
    return results


def main():
    """main/driver function."""

    # ==================================================
    # STEP ONE: PARSE CLI

    # check for sufficient arguments
    if (len(sys.argv) < 6):
        print('USAGE: vectorspace.py DOC_WEIGHT_MODE QUERY_WEIGHT_MODE COSINE_NORM(T/F) DOCUMENT_DIR TEST_QUERY_FILE (VERBOSE(T/F))')
        raise ValueError(f"Expected at least {5} arguments, got {len(sys.argv)}.")

    # argument 6: verbose flag
    verbose = False
    if (len(sys.argv) == 7):
        verbose = (sys.argv[6] == 'T' | sys.argv[6] == 'True')

    # check for extra arguments
    if (len(sys.argv) > 7):
        print('USAGE: vectorspace.py DOC_WEIGHT_MODE QUERY_WEIGHT_MODE COSINE_NORM(T/F) DOCUMENT_DIR TEST_QUERY_FILE (VERBOSE(T/F))')
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
    doc_dir_path = pathlib.Path(sys.argv[4])
    if not doc_dir_path.is_dir():
        FileNotFoundError(f'The directory {sys.argv[4]} does not exist.')
    elif (verbose):
        print('document directory found successfully.')

    # argument 5: test query file
    test_query_filepath = pathlib.Path(sys.argv[5])
    if not test_query_filepath.is_file():
        FileNotFoundError(f'The file {sys.argv[5]} does not exist.')
    elif (verbose):
        print('test query file found successfully.')

    # ==================================================
    # STEP TWO: TBA


if __name__ == "__main__":
    main()