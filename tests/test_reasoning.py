from iot_security.pipeline import CollaborativeReasoningLayer


def test_reasoning_votes_above_threshold():
    layer = CollaborativeReasoningLayer(n_agents=3, threshold=0.6)
    results = [
        {"verdict": "ATTACK", "confidence": 0.9, "attack_type": "DDoS"},
        {"verdict": "ATTACK", "confidence": 0.8, "attack_type": "DDoS"},
        {"verdict": "BENIGN", "confidence": 0.7, "attack_type": "None"},
    ]
    consensus = layer.vote(results)
    assert consensus["decision"] == "THREAT_CONFIRMED"
    assert consensus["attack_type"] == "DDoS"


def test_reasoning_votes_below_threshold():
    layer = CollaborativeReasoningLayer(n_agents=3, threshold=0.7)
    results = [
        {"verdict": "ATTACK", "confidence": 0.9, "attack_type": "DDoS"},
        {"verdict": "BENIGN", "confidence": 0.8, "attack_type": "None"},
        {"verdict": "BENIGN", "confidence": 0.7, "attack_type": "None"},
    ]
    consensus = layer.vote(results)
    assert consensus["decision"] == "BENIGN"
    assert consensus["action"] == "ALLOW"
