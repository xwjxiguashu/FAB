"""Random FAB scheduling problem generation for PPO training."""

from dataclasses import dataclass

import numpy as np

from problem_instances import Phase1CalendarProblem


@dataclass(frozen=True)
class RandomProblemConfig:
    num_lots: int
    num_machines: int
    num_ppids_per_machine: int
    min_stages: int
    max_stages: int
    num_chambers: int
    num_sides: int
    min_candidate_resources: int
    max_candidate_resources: int
    min_process_time: float
    max_process_time: float
    min_wafers: int
    max_wafers: int
    arrival_span: float
    due_tightness: float
    qtime_probability: float
    qtime_tightness: float
    machine_eligibility_ratio: float
    recipe_count: int
    seed: int
    difficulty: str = "custom"


def build_easy_config(seed):
    return RandomProblemConfig(
        num_lots=6,
        num_machines=2,
        num_ppids_per_machine=1,
        min_stages=2,
        max_stages=3,
        num_chambers=2,
        num_sides=2,
        min_candidate_resources=1,
        max_candidate_resources=2,
        min_process_time=1.5,
        max_process_time=4.0,
        min_wafers=2,
        max_wafers=4,
        arrival_span=6.0,
        due_tightness=4.0,
        qtime_probability=0.25,
        qtime_tightness=3.0,
        machine_eligibility_ratio=1.0,
        recipe_count=3,
        seed=int(seed),
        difficulty="easy",
    )


def build_medium_config(seed):
    return RandomProblemConfig(
        num_lots=14,
        num_machines=4,
        num_ppids_per_machine=2,
        min_stages=2,
        max_stages=4,
        num_chambers=3,
        num_sides=2,
        min_candidate_resources=1,
        max_candidate_resources=3,
        min_process_time=1.5,
        max_process_time=5.0,
        min_wafers=3,
        max_wafers=6,
        arrival_span=12.0,
        due_tightness=2.8,
        qtime_probability=0.45,
        qtime_tightness=2.2,
        machine_eligibility_ratio=0.75,
        recipe_count=5,
        seed=int(seed),
        difficulty="medium",
    )


def build_hard_config(seed):
    return RandomProblemConfig(
        num_lots=28,
        num_machines=7,
        num_ppids_per_machine=3,
        min_stages=3,
        max_stages=5,
        num_chambers=4,
        num_sides=2,
        min_candidate_resources=1,
        max_candidate_resources=3,
        min_process_time=1.5,
        max_process_time=6.0,
        min_wafers=4,
        max_wafers=8,
        arrival_span=18.0,
        due_tightness=1.9,
        qtime_probability=0.65,
        qtime_tightness=1.6,
        machine_eligibility_ratio=0.55,
        recipe_count=7,
        seed=int(seed),
        difficulty="hard",
    )


def sample_random_problem_config(episode, split="train"):
    split_offsets = {
        "train": 0,
        "validation": 10000,
        "test": 20000,
    }
    if split not in split_offsets:
        raise ValueError("split must be one of: train, validation, test")

    episode = int(episode)
    seed = split_offsets[split] + episode
    if episode < 100:
        return build_easy_config(seed)
    if episode < 200:
        return build_medium_config(seed)
    return build_hard_config(seed)


def _choose_feasible_machines(rng, num_machines, eligibility_ratio):
    machines = list(range(1, int(num_machines) + 1))
    selected = [
        machine
        for machine in machines
        if float(rng.random()) <= float(eligibility_ratio)
    ]
    if selected:
        return selected
    return [int(rng.choice(machines))]


def _stage_resources(rng, config):
    resource_count = int(
        rng.integers(
            int(config.min_candidate_resources),
            int(config.max_candidate_resources) + 1,
        )
    )
    all_resources = [
        (chamber, side)
        for chamber in range(1, int(config.num_chambers) + 1)
        for side in range(int(config.num_sides))
    ]
    resource_count = min(resource_count, len(all_resources))
    chosen_indices = rng.choice(len(all_resources), size=resource_count, replace=False)
    rows = []
    for index in chosen_indices:
        chamber, side = all_resources[int(index)]
        process_time = float(rng.uniform(config.min_process_time, config.max_process_time))
        rows.append([chamber, side, process_time])
    return np.asarray(rows, dtype=float)


def _estimate_nominal_duration(ppid_steps, feasible_ppids, feasible_machines, wafer_counts, lot):
    durations = []
    for machine in feasible_machines[lot]:
        for ppid in feasible_ppids[(lot, machine)]:
            stage_time = sum(float(np.min(np.asarray(stage, dtype=float)[:, 2])) for stage in ppid_steps[(lot, machine, ppid)])
            durations.append(stage_time * float(wafer_counts[lot]))
    return max(min(durations), 1.0)


def build_random_encoder(config):
    rng = np.random.default_rng(int(config.seed))
    num_lots = int(config.num_lots)
    num_machines = int(config.num_machines)

    wafer_counts = {
        lot: int(rng.integers(int(config.min_wafers), int(config.max_wafers) + 1))
        for lot in range(1, num_lots + 1)
    }
    feasible_machines = {
        lot: _choose_feasible_machines(rng, num_machines, config.machine_eligibility_ratio)
        for lot in range(1, num_lots + 1)
    }

    feasible_ppids = {}
    ppid_steps = {}
    q_time_limits = {}
    for lot in range(1, num_lots + 1):
        for machine in feasible_machines[lot]:
            ppids = [
                lot * 10000 + machine * 100 + ppid_index
                for ppid_index in range(1, int(config.num_ppids_per_machine) + 1)
            ]
            feasible_ppids[(lot, machine)] = ppids
            for ppid in ppids:
                stage_count = int(rng.integers(int(config.min_stages), int(config.max_stages) + 1))
                steps = [_stage_resources(rng, config) for _ in range(stage_count)]
                ppid_steps[(lot, machine, ppid)] = steps
                for from_stage in range(1, stage_count):
                    if float(rng.random()) <= float(config.qtime_probability):
                        previous_time = float(np.min(steps[from_stage - 1][:, 2]))
                        next_time = float(np.min(steps[from_stage][:, 2]))
                        q_time_limits[(lot, machine, ppid, from_stage, from_stage + 1)] = (
                            previous_time + next_time * float(config.qtime_tightness)
                        )

    arrival_times = {
        lot: float(rng.uniform(0.0, max(float(config.arrival_span), 0.0)))
        for lot in range(1, num_lots + 1)
    }
    arrival_times[1] = 0.0
    priorities = {
        lot: float(rng.uniform(0.0, 10.0))
        for lot in range(1, num_lots + 1)
    }
    recipe = {
        lot: f"R{1 + ((lot - 1) % int(config.recipe_count))}"
        for lot in range(1, num_lots + 1)
    }
    due_dates = {}
    for lot in range(1, num_lots + 1):
        nominal_duration = _estimate_nominal_duration(
            ppid_steps,
            feasible_ppids,
            feasible_machines,
            wafer_counts,
            lot,
        )
        due_dates[lot] = float(arrival_times[lot] + nominal_duration * float(config.due_tightness))

    encoder = Phase1CalendarProblem(
        num_lots=num_lots,
        num_machines=num_machines,
        feasible_machines=feasible_machines,
        feasible_ppids=feasible_ppids,
        ppid_steps=ppid_steps,
        wafer_counts=wafer_counts,
        due_dates=due_dates,
        priorities=priorities,
        q_time_limits=q_time_limits,
        recipe=recipe,
        machine_group={
            machine: f"G{1 + ((machine - 1) // max(1, min(5, num_machines)))}"
            for machine in range(1, num_machines + 1)
        },
        machine_resources={
            machine: [
                (chamber, side)
                for chamber in range(1, int(config.num_chambers) + 1)
                for side in range(int(config.num_sides))
            ]
            for machine in range(1, num_machines + 1)
        },
    )
    encoder.arrival_times = arrival_times
    encoder.validate_problem_definition()
    return encoder
