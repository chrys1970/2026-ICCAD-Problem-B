import argparse
import csv
import numpy as np
import time  # <-- Import the time module
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from feature_extraction import extract_regr_features, extract_sim_features, extract_trace_features
from feature_encoding import encode_features_to_array

def main():
    # Start the high-resolution timer
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(description="Regression Failure Bucketing")
    parser.add_argument("--input", required=True, help="Path to input.csv")
    parser.add_argument("--output", required=True, help="Path to save output.csv")
    parser.add_argument("--k", type=int, required=True, help="Hint for number of clusters")
    args = parser.parse_args()

    cases = []
    print(f"Reading data from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 4:
                trace_log, sim_log, regr_log, case_id = row
                cases.append({
                    "trace": trace_log,
                    "sim": sim_log,
                    "regr": regr_log,
                    "case_id": case_id
                })

    print(f"Extracting and encoding features for {len(cases)} cases...")
    feature_vectors = []
    
    for case in cases:
        regr_feat = extract_regr_features(case["regr"])
        sim_feat = extract_sim_features(case["sim"])
        trace_feat = extract_trace_features(case["trace"])
        
        vector = encode_features_to_array(regr_feat, sim_feat, trace_feat)
        feature_vectors.append(vector)

    X = np.array(feature_vectors)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Running Agglomerative Clustering with K={args.k}...")
    clusterer = AgglomerativeClustering(n_clusters=args.k)
    cluster_labels = clusterer.fit_predict(X_scaled)

    print(f"Saving results to {args.output}...")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for label in cluster_labels:
            writer.writerow([label])

    # Stop the timer and calculate the difference
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    print("Pipeline Complete!")
    print(f"Total Execution Time: {execution_time:.4f} seconds")

if __name__ == "__main__":
    main()