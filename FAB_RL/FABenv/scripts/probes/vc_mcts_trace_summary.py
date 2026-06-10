"""Summarize VC-MCTS decision traces written as JSONL."""

from pathlib import Path
import sys

FABENV_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = FABENV_ROOT / "scripts"
for path in (
    FABENV_ROOT,
    SCRIPT_ROOT / "run",
    SCRIPT_ROOT / "evaluation",
    SCRIPT_ROOT / "experiments",
    SCRIPT_ROOT / "probes",
):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
import argparse
import json
from pathlib import Path


def read_jsonl_trace(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _edge_values(edges, kind, metric):
    values = []
    for edge in edges:
        if edge.get("kind") != kind:
            continue
        value = edge.get(metric)
        if value is not None:
            values.append(float(value))
    return values


def _best_metric(edges, kind, metric):
    values = _edge_values(edges, kind, metric)
    return None if not values else min(values)


def _best_non_reserve(edges, metric):
    by_kind = {
        kind: _best_metric(edges, kind, metric)
        for kind in ("dispatch", "delegate_dispatch", "no_op")
    }
    candidates = {
        kind: value
        for kind, value in by_kind.items()
        if value is not None
    }
    if not candidates:
        return None, None
    kind, value = min(candidates.items(), key=lambda item: (item[1], item[0]))
    return kind, value


def _avg(values):
    return None if not values else sum(values) / len(values)


def summarize_trace_records(records):
    selected_counts = {}
    reserve_available_decisions = 0
    reserve_selected_decisions = 0
    reserve_lost_decisions = 0
    reserve_lost_to_counts = {}
    reserve_total_visits = 0
    reserve_edge_count_total = 0
    reserve_o2_gaps = []
    reserve_qtime_gaps = []
    reserve_qtime_total_gaps = []
    reserve_o2_worse_count = 0
    reserve_qtime_worse_count = 0
    reserve_qtime_better_count = 0
    reserve_qtime_total_worse_count = 0
    reserve_qtime_total_better_count = 0
    selected_reserve_lots = []
    first_time = None
    last_time = None
    # 机制 2 (报告8 §7.12.4): Δρ_pc 对冲水位诊断
    rho_deltas = []
    rho_positive_delta_edges = 0
    selected_reserve_deltas = []

    for record in records:
        edges = list(record.get("edges", []))
        diagnostics = record.get("diagnostics", {})
        selected_action = record.get("selected_action", {})
        selected_kind = (
            selected_action.get("kind")
            or diagnostics.get("selected_kind")
            or "unknown"
        )
        selected_counts[selected_kind] = selected_counts.get(selected_kind, 0) + 1

        if "time" in record:
            current_time = float(record["time"])
            first_time = current_time if first_time is None else min(first_time, current_time)
            last_time = current_time if last_time is None else max(last_time, current_time)

        reserve_edges = [edge for edge in edges if edge.get("kind") == "reserve"]
        reserve_edge_count_total += len(reserve_edges)
        reserve_total_visits += sum(int(edge.get("visits") or 0) for edge in reserve_edges)

        for edge in edges:
            if "delta_rho_pc" not in edge:
                continue
            delta = float(edge["delta_rho_pc"])
            rho_deltas.append(delta)
            if delta > 0.0:
                rho_positive_delta_edges += 1
            if selected_kind == "reserve" and edge.get("kind") == "reserve":
                selected_reserve_deltas.append(delta)

        reserve_available = bool(
            diagnostics.get("reserve_was_available", bool(reserve_edges))
        )
        reserve_selected = bool(
            diagnostics.get("reserve_was_selected", selected_kind == "reserve")
        )
        if not reserve_available:
            continue

        reserve_available_decisions += 1
        if reserve_selected:
            reserve_selected_decisions += 1
            future_lot = selected_action.get("future_lot")
            if future_lot is not None:
                selected_reserve_lots.append(int(future_lot))
        else:
            reserve_lost_decisions += 1
            winner_kind = selected_kind or "unknown"
            reserve_lost_to_counts[winner_kind] = (
                reserve_lost_to_counts.get(winner_kind, 0) + 1
            )

        best_reserve_o2 = diagnostics.get("best_reserve_o2")
        if best_reserve_o2 is None:
            best_reserve_o2 = _best_metric(edges, "reserve", "mean_o2")
        _, best_non_reserve_o2 = _best_non_reserve(edges, "mean_o2")
        if best_reserve_o2 is not None and best_non_reserve_o2 is not None:
            o2_gap = float(best_reserve_o2) - float(best_non_reserve_o2)
            reserve_o2_gaps.append(o2_gap)
            if o2_gap > 0.0:
                reserve_o2_worse_count += 1

        best_reserve_qtime = _best_metric(edges, "reserve", "mean_qtime")
        _, best_non_reserve_qtime = _best_non_reserve(edges, "mean_qtime")
        if best_reserve_qtime is not None and best_non_reserve_qtime is not None:
            qtime_gap = float(best_reserve_qtime) - float(best_non_reserve_qtime)
            reserve_qtime_gaps.append(qtime_gap)
            if qtime_gap > 0.0:
                reserve_qtime_worse_count += 1
            elif qtime_gap < 0.0:
                reserve_qtime_better_count += 1

        best_reserve_qtime_total = _best_metric(edges, "reserve", "mean_qtime_total")
        _, best_non_reserve_qtime_total = _best_non_reserve(edges, "mean_qtime_total")
        if (
            best_reserve_qtime_total is not None
            and best_non_reserve_qtime_total is not None
        ):
            qtime_total_gap = (
                float(best_reserve_qtime_total)
                - float(best_non_reserve_qtime_total)
            )
            reserve_qtime_total_gaps.append(qtime_total_gap)
            if qtime_total_gap > 0.0:
                reserve_qtime_total_worse_count += 1
            elif qtime_total_gap < 0.0:
                reserve_qtime_total_better_count += 1

    decisions = len(records)
    duplicate_selected_reserve_lots = sorted(
        lot
        for lot in set(selected_reserve_lots)
        if selected_reserve_lots.count(lot) > 1
    )
    return {
        "decisions": int(decisions),
        "first_time": first_time,
        "last_time": last_time,
        "selected_counts": selected_counts,
        "reserve_available_decisions": int(reserve_available_decisions),
        "reserve_selected_decisions": int(reserve_selected_decisions),
        "reserve_selection_rate_when_available": (
            None
            if reserve_available_decisions == 0
            else reserve_selected_decisions / reserve_available_decisions
        ),
        "reserve_total_visits": int(reserve_total_visits),
        "reserve_edge_count_total": int(reserve_edge_count_total),
        "reserve_lost_decisions": int(reserve_lost_decisions),
        "reserve_lost_to_counts": reserve_lost_to_counts,
        "reserve_o2_gap_vs_best_non_reserve_avg": _avg(reserve_o2_gaps),
        "reserve_qtime_gap_vs_best_non_reserve_avg": _avg(reserve_qtime_gaps),
        "reserve_qtime_total_gap_vs_best_non_reserve_avg": _avg(reserve_qtime_total_gaps),
        "reserve_o2_worse_count": int(reserve_o2_worse_count),
        "reserve_qtime_worse_count": int(reserve_qtime_worse_count),
        "reserve_qtime_better_count": int(reserve_qtime_better_count),
        "reserve_qtime_total_worse_count": int(reserve_qtime_total_worse_count),
        "reserve_qtime_total_better_count": int(reserve_qtime_total_better_count),
        "selected_reserve_lots": selected_reserve_lots,
        "duplicate_selected_reserve_lots": duplicate_selected_reserve_lots,
        "rho_pc_edge_count": int(len(rho_deltas)),
        "rho_pc_positive_delta_edges": int(rho_positive_delta_edges),
        "rho_pc_delta_avg": _avg(rho_deltas),
        "rho_pc_selected_reserve_delta_avg": _avg(selected_reserve_deltas),
    }


def summarize_trace_file(path):
    return summarize_trace_records(read_jsonl_trace(path))


def main(trace_path, out=None):
    summary = summarize_trace_file(trace_path)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return summary


def _cli():
    parser = argparse.ArgumentParser(description="Summarize a VC-MCTS trace JSONL file")
    parser.add_argument("trace_path")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    main(args.trace_path, out=args.out)


if __name__ == "__main__":
    _cli()
