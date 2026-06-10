import json

from vc_mcts_trace_summary import read_jsonl_trace, summarize_trace_records


def _edge(kind, mean_qtime, mean_o2, visits=1, mean_qtime_total=0.0):
    return {
        "action": {"kind": kind},
        "kind": kind,
        "visits": visits,
        "mean_qtime": mean_qtime,
        "mean_qtime_total": mean_qtime_total,
        "mean_o2": mean_o2,
        "mean_util": 0.5,
    }


def test_summarize_trace_records_counts_reserve_availability_and_losses():
    records = [
        {
            "time": 0.0,
            "machine": 1,
            "selected_action": {"kind": "dispatch"},
            "edges": [
                _edge("no_op", 2.0, 12.0, mean_qtime_total=4.0),
                _edge("dispatch", 1.0, 10.0, mean_qtime_total=2.0),
                _edge("reserve", 3.0, 15.0, visits=2, mean_qtime_total=5.0),
            ],
            "diagnostics": {"reserve_was_available": True, "reserve_was_selected": False},
        },
        {
            "time": 1.0,
            "machine": 1,
            "selected_action": {"kind": "reserve", "future_lot": 7},
            "edges": [
                _edge("no_op", 2.0, 12.0, mean_qtime_total=2.0),
                _edge("dispatch", 2.0, 13.0, mean_qtime_total=3.0),
                _edge("reserve", 0.0, 8.0, mean_qtime_total=0.0),
            ],
            "diagnostics": {"reserve_was_available": True, "reserve_was_selected": True},
        },
        {
            "time": 2.0,
            "machine": 2,
            "selected_action": {"kind": "reserve", "future_lot": 7},
            "edges": [
                _edge("no_op", 1.0, 5.0),
                _edge("reserve", 1.0, 4.0),
            ],
            "diagnostics": {"reserve_was_available": True, "reserve_was_selected": True},
        },
        {
            "time": 3.0,
            "machine": 2,
            "selected_action": {"kind": "no_op"},
            "edges": [_edge("no_op", 1.0, 5.0)],
            "diagnostics": {"reserve_was_available": False, "reserve_was_selected": False},
        },
    ]

    summary = summarize_trace_records(records)

    assert summary["decisions"] == 4
    assert summary["selected_counts"] == {"dispatch": 1, "reserve": 2, "no_op": 1}
    assert summary["reserve_available_decisions"] == 3
    assert summary["reserve_selected_decisions"] == 2
    assert summary["reserve_selection_rate_when_available"] == 2 / 3
    assert summary["reserve_total_visits"] == 4
    assert summary["reserve_lost_decisions"] == 1
    assert summary["reserve_lost_to_counts"] == {"dispatch": 1}
    assert summary["reserve_o2_gap_vs_best_non_reserve_avg"] == 0.0
    assert summary["reserve_qtime_gap_vs_best_non_reserve_avg"] == 0.0
    assert summary["reserve_qtime_total_gap_vs_best_non_reserve_avg"] == 1 / 3
    assert summary["selected_reserve_lots"] == [7, 7]
    assert summary["duplicate_selected_reserve_lots"] == [7]


def test_reserve_lost_to_counts_uses_final_selected_action_after_noop_gating():
    records = [
        {
            "time": 0.0,
            "machine": 1,
            "selected_action": {"kind": "delegate_dispatch"},
            "edges": [
                _edge("no_op", 0.0, 1.0, mean_qtime_total=0.0),
                _edge("delegate_dispatch", 0.0, 8.0, mean_qtime_total=0.0),
                _edge("reserve", 1.0, 10.0, visits=2, mean_qtime_total=1.0),
            ],
            "diagnostics": {
                "reserve_was_available": True,
                "reserve_was_selected": False,
                "selected_kind": "delegate_dispatch",
            },
        },
    ]

    summary = summarize_trace_records(records)

    assert summary["selected_counts"] == {"delegate_dispatch": 1}
    assert summary["reserve_lost_decisions"] == 1
    assert summary["reserve_lost_to_counts"] == {"delegate_dispatch": 1}


def test_delegate_dispatch_is_included_in_best_non_reserve_gaps():
    records = [
        {
            "time": 0.0,
            "machine": 1,
            "selected_action": {"kind": "reserve", "future_lot": 8},
            "edges": [
                _edge("no_op", 0.0, 20.0, mean_qtime_total=0.0),
                _edge("delegate_dispatch", 0.0, 7.0, mean_qtime_total=0.0),
                _edge("reserve", 0.0, 10.0, visits=2, mean_qtime_total=0.0),
            ],
            "diagnostics": {"reserve_was_available": True, "reserve_was_selected": True},
        },
    ]

    summary = summarize_trace_records(records)

    assert summary["reserve_o2_gap_vs_best_non_reserve_avg"] == 3.0


def test_read_jsonl_trace_skips_blank_lines(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps({"selected_action": {"kind": "no_op"}, "edges": []})
        + "\n\n"
        + json.dumps({"selected_action": {"kind": "reserve"}, "edges": []})
        + "\n",
        encoding="utf-8",
    )

    records = read_jsonl_trace(trace_path)

    assert [record["selected_action"]["kind"] for record in records] == ["no_op", "reserve"]


def test_trace_summary_reports_delta_rho_pc_buckets(tmp_path):
    from vc_mcts_trace_summary import summarize_trace_file

    path = tmp_path / "trace.jsonl"
    rows = [
        {
            "selected_action": {"kind": "reserve", "future_lot": 3},
            "edges": [
                {"kind": "reserve", "delta_rho_pc": 0.50, "mean_o2": 8.0},
                {"kind": "delegate_dispatch", "delta_rho_pc": -0.25, "mean_o2": 10.0},
            ],
            "diagnostics": {"reserve_was_available": True, "reserve_was_selected": True},
        },
        {
            "selected_action": {"kind": "delegate_dispatch"},
            "edges": [
                {"kind": "reserve", "delta_rho_pc": 0.10, "mean_o2": 11.0},
                {"kind": "delegate_dispatch", "delta_rho_pc": 0.0, "mean_o2": 9.0},
            ],
            "diagnostics": {"reserve_was_available": True, "reserve_was_selected": False},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    summary = summarize_trace_file(path)

    assert summary["rho_pc_edge_count"] == 4
    assert summary["rho_pc_positive_delta_edges"] == 2
    assert summary["rho_pc_selected_reserve_delta_avg"] == 0.5
