from __future__ import annotations

import pandas as pd

from fedcrg.datasets.diad import DIAD_FEATURES, DiadAdapter


def test_binary_anomaly_label_and_attack_category_are_distinct_contracts() -> None:
    frame = pd.DataFrame(columns=["device_mac", "label", "attack_category", *DIAD_FEATURES])
    anomaly = DiadAdapter._anomaly_label_column(frame)
    category = DiadAdapter._attack_category_column(frame, anomaly)
    assert anomaly == "label"
    assert category == "attack_category"
    assert anomaly != category


def test_identity_labels_ports_and_category_are_not_model_features() -> None:
    forbidden = {
        "device_mac",
        "label",
        "attack_category",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "port_class_dst",
        "stream",
    }
    assert forbidden.isdisjoint(DIAD_FEATURES)
    assert len(DIAD_FEATURES) == 86
