from pathlib import Path
import argparse
import json

from src.iot_security.metrics import evaluate_results_csv
from src.iot_security.pipeline import (
    CollaborativeReasoningLayer,
    EdgeAgent,
    MitigationAgent,
    run_multi_agent_simulation,
)
from src.iot_security.preprocessing import load_and_clean_dataset
from src.iot_security.training import run_training_pipeline


def run(args):
    root = Path(__file__).resolve().parent
    data_csv = root / "data" / "my_iot_dataset.csv"
    processed_dir = root / "data" / "processed"
    models_dir = root / "models"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    trained = run_training_pipeline(data_csv, processed_dir, models_dir)

    df = load_and_clean_dataset(data_csv)
    min_class_size = int(df["category"].value_counts().min())
    if args.sample_per_class > min_class_size:
        raise ValueError(
            f"--sample-per-class={args.sample_per_class} exceeds smallest class size ({min_class_size})."
        )

    sample_df = (
        df.groupby("category", group_keys=False)
        .sample(n=args.sample_per_class, random_state=args.seed, replace=False)
        .reset_index(drop=True)
    )

    agents = [
        EdgeAgent(
            agent_id=i + 1,
            cuckoo_filter=trained["cuckoo_filter"],
            iso_model=trained["isolation_forest"],
            rf_model=trained["random_forest"],
            scaler=trained["scaler"],
            label_encoder=trained["label_encoder"],
        )
        for i in range(args.agents)
    ]
    reasoner = CollaborativeReasoningLayer(n_agents=args.agents, threshold=args.threshold)
    mitigator = MitigationAgent()
    sim_df = run_multi_agent_simulation(sample_df, agents, reasoner, mitigator)

    results_csv = results_dir / "multi_agent_simulation.csv"
    sim_df.to_csv(results_csv, index=False)

    metrics = evaluate_results_csv(results_csv)
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Pipeline complete.")
    print(f"Saved simulation: {results_csv}")
    print(f"Saved metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Run IoT edge security pipeline end-to-end.")
    parser.add_argument("--agents", type=int, default=3, help="Number of edge agents in simulation.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Consensus threshold.")
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=40,
        help="Packets per category for simulation output.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
