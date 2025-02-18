# Paul Vander Woude (pavander)

# Evaluation functions for generating .answer file(s)

import random
from collections import defaultdict

def load_reljudge(reljudge_filepath, verbose=0):
    """Load relevance judgments from a file."""
    reljudge = {}
    with open(reljudge_filepath, "r", encoding="ISO-8859-1") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            qid = int(parts[0])
            doc_id = int(parts[1])
            reljudge.setdefault(qid, []).append(doc_id)
    if verbose >= 1:
        print(f"[load_reljudge] Loaded relevance judgments for {len(reljudge)} queries.")
    return reljudge


def evaluate_query(query_id, retrieved_docs, reljudge, all_doc_ids, cutoffs=(10, 50, 100, 500)):
    """Compute precision and recall for a single query at various cutoffs."""
    # Get the set of relevant document IDs for this query
    relevant = set(reljudge.get(query_id, []))
    total_rel = len(relevant)
    results = []
    
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
        results.append((cutoff, precision, recall))
    return results


def average_precision_recall(data):
    """Computes the average precision and recall for each unique cutoff value."""
    cutoff_stats = defaultdict(lambda: {'precision_sum': 0, 'recall_sum': 0, 'count': 0})
    
    for cutoff, precision, recall in data:
        cutoff_stats[cutoff]['precision_sum'] += precision
        cutoff_stats[cutoff]['recall_sum'] += recall
        cutoff_stats[cutoff]['count'] += 1
    
    avg_results = {
        cutoff: (
            stats['precision_sum'] / stats['count'],
            stats['recall_sum'] / stats['count']
        )
        for cutoff, stats in cutoff_stats.items()
    }
    
    return avg_results


def write_evaluation(answers_filepath, eval_results, header):
    with open(answers_filepath, "w", encoding="ISO-8859-1") as ans_file:
        # First line: configuration info.
        ans_file.write(header)
        avg_results = average_precision_recall(eval_results)
        for cutoff, val in avg_results.items():
            ans_file.write(f"cutoff: {cutoff} precision: {val[0]:.4f} recall: {val[1]:.4f}\n")