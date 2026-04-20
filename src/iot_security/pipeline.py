from collections import Counter
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from .constants import FEATURE_COLS, LABEL_COL


def make_signature(row: pd.Series) -> str:
    return f"{round(row['Rate'], 1)}_{row['Protocol Type']}_{row['ICMP']}"


class EdgeAgent:
    def __init__(self, agent_id: int, cuckoo_filter, iso_model, rf_model, scaler, label_encoder):
        self.agent_id = agent_id
        self.cuckoo_filter = cuckoo_filter
        self.iso_model = iso_model
        self.rf_model = rf_model
        self.scaler = scaler
        self.label_encoder = label_encoder

    def analyze(self, row: pd.Series) -> Dict[str, object]:
        start = time.perf_counter()
        signature = make_signature(row)

        if self.cuckoo_filter.lookup(signature):
            verdict = "ATTACK"
            confidence = 0.95
            stage = "cuckoo_filter"
            attack_type = "Known_Threat"
        else:
            features = np.array(row[FEATURE_COLS].values, dtype=np.float32).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            iso_pred = self.iso_model.predict(features_scaled)[0]

            if iso_pred == -1:
                verdict = "ATTACK"
                confidence = 0.75
                stage = "isolation_forest"
                attack_type = "Unknown_Anomaly"
            else:
                rf_pred = self.rf_model.predict(features_scaled)[0]
                rf_proba = self.rf_model.predict_proba(features_scaled)[0]
                label = self.label_encoder.classes_[rf_pred]
                verdict = "ATTACK" if label != "Benign" else "BENIGN"
                confidence = float(np.max(rf_proba))
                stage = "random_forest"
                attack_type = str(label)

        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "agent_id": self.agent_id,
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "stage": stage,
            "attack_type": attack_type,
            "latency_ms": round(latency_ms, 4),
        }


class CollaborativeReasoningLayer:
    def __init__(self, n_agents: int, threshold: float = 0.6):
        self.n_agents = n_agents
        self.threshold = threshold

    def vote(self, agent_results: List[Dict[str, object]]) -> Dict[str, object]:
        attack_results = [r for r in agent_results if r["verdict"] == "ATTACK"]
        attack_votes = len(attack_results)
        threat_ratio = attack_votes / self.n_agents
        avg_conf = float(np.mean([r["confidence"] for r in agent_results]))
        weighted_conf = round(avg_conf * threat_ratio, 3)

        if threat_ratio >= self.threshold:
            attack_type = Counter([r["attack_type"] for r in attack_results]).most_common(1)[0][0]
            return {
                "decision": "THREAT_CONFIRMED",
                "threat_ratio": round(threat_ratio, 2),
                "weighted_conf": weighted_conf,
                "attack_type": attack_type,
                "action": "MITIGATE",
                "votes": f"{attack_votes}/{self.n_agents}",
            }

        return {
            "decision": "BENIGN",
            "threat_ratio": round(threat_ratio, 2),
            "weighted_conf": weighted_conf,
            "attack_type": "None",
            "action": "ALLOW",
            "votes": f"{attack_votes}/{self.n_agents}",
        }


class MitigationAgent:
    def respond(self, consensus: Dict[str, object], packet_idx: int) -> Dict[str, object]:
        if consensus["action"] == "MITIGATE":
            return {
                "packet_idx": packet_idx,
                "action": "ISOLATE_AND_BLOCK",
                "attack_type": consensus["attack_type"],
                "confidence": consensus["weighted_conf"],
                "votes": consensus["votes"],
                "xai_reason": (
                    f"{consensus['votes']} agents confirmed threat. "
                    f"Type: {consensus['attack_type']}. "
                    f"Weighted confidence: {consensus['weighted_conf']}. "
                    "Action: device isolated, firewall rule applied."
                ),
            }

        return {
            "packet_idx": packet_idx,
            "action": "ALLOW",
            "attack_type": "None",
            "confidence": round(1.0 - consensus["threat_ratio"], 3),
            "votes": consensus["votes"],
            "xai_reason": (
                f"Only {consensus['votes']} agents flagged threat. "
                f"Below {int(consensus['threat_ratio'] * 100)}% threshold. "
                "Traffic allowed."
            ),
        }


def run_multi_agent_simulation(
    df: pd.DataFrame,
    agents: List[EdgeAgent],
    reasoner: CollaborativeReasoningLayer,
    mitigator: MitigationAgent,
) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        agent_results = [agent.analyze(row) for agent in agents]
        consensus = reasoner.vote(agent_results)
        action = mitigator.respond(consensus, idx)

        true_label = str(row[LABEL_COL])
        true_binary = 0 if true_label == "Benign" else 1
        pred_binary = 1 if consensus["decision"] == "THREAT_CONFIRMED" else 0

        rows.append(
            {
                "packet_idx": idx,
                "true_label": true_label,
                "decision": consensus["decision"],
                "votes": consensus["votes"],
                "action": action["action"],
                "attack_type": action["attack_type"],
                "confidence": action["confidence"],
                "xai_reason": action["xai_reason"],
                "true_binary": true_binary,
                "pred_binary": pred_binary,
                "latency_ms": round(sum(r["latency_ms"] for r in agent_results), 4),
            }
        )
    return pd.DataFrame(rows)
