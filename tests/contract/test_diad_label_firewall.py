from __future__ import annotations

from fedcrg.data.datasets import DIAD_FEATURES


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


def test_feature_contract_is_frozen() -> None:
    assert DIAD_FEATURES[:11] == (
        "inter_arrival_time",
        "time_since_previously_displayed_frame",
        "l4_tcp",
        "l4_udp",
        "ttl",
        "eth_size",
        "tcp_window_size",
        "payload_entropy",
        "payload_length",
        "l3_ip_dst_count",
        "jitter",
    )
    assert DIAD_FEATURES[11] == "stream_1_count"
    assert DIAD_FEATURES[-1] == "stream_jitter_60_var"
