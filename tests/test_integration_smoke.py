from pathlib import Path
import tempfile

import pandas as pd

from iot_security.metrics import evaluate_results_csv
from iot_security.pipeline import (
    CollaborativeReasoningLayer,
    EdgeAgent,
    MitigationAgent,
    run_multi_agent_simulation,
)
from iot_security.preprocessing import load_and_clean_dataset
from iot_security.training import run_training_pipeline


def test_end_to_end_smoke():
    root = Path(__file__).resolve().parents[1]
    data_csv = root / "data" / "my_iot_dataset.csv"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        processed_dir = tmp_path / "processed"
        models_dir = tmp_path / "models"
        results_csv = tmp_path / "results.csv"

        tiny_csv = tmp_path / "tiny.csv"
        full_df = load_and_clean_dataset(data_csv)
        tiny_df = (
            full_df.groupby("category", group_keys=False)
            .head(40)
            .reset_index(drop=True)
        )
        tiny_df.to_csv(tiny_csv, index=False)

        trained = run_training_pipeline(tiny_csv, processed_dir, models_dir)
        df = tiny_df.groupby("category", group_keys=False).head(3).reset_index(drop=True)

        agents = [
            EdgeAgent(
                agent_id=i + 1,
                cuckoo_filter=trained["cuckoo_filter"],
                iso_model=trained["isolation_forest"],
                rf_model=trained["random_forest"],
                scaler=trained["scaler"],
                label_encoder=trained["label_encoder"],
            )
            for i in range(3)
        ]
        reasoner = CollaborativeReasoningLayer(n_agents=3, threshold=0.6)
        mitigator = MitigationAgent()
        sim_df = run_multi_agent_simulation(df, agents, reasoner, mitigator)

        sim_df.to_csv(results_csv, index=False)
        metrics = evaluate_results_csv(results_csv)

        assert isinstance(sim_df, pd.DataFrame)
        assert len(sim_df) > 0
        assert 0.0 <= metrics["accuracy"] <= 1.0
