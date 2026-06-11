from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FABENV = ROOT / "FAB_RL" / "FABenv"


def test_root_level_materials_are_classified():
    expected_paths = [
        ROOT / "docs" / "project" / "项目方案.md",
        ROOT / "docs" / "project" / "项目建模说明.md",
        ROOT / "docs" / "materials" / "FAB项目.docx",
        ROOT / "docs" / "materials" / "FAB厂机台组调度问题说明.pptx",
        ROOT / "docs" / "reports" / "项目报告7.md",
        ROOT / "docs" / "reports" / "项目报告8.md",
        ROOT / "docs" / "reports" / "mechanism_results",
        ROOT / "docs" / "reports" / "archive" / "报告.md",
        ROOT / "docs" / "reports" / "archive" / "项目报告.md",
        ROOT / "docs" / "reports" / "archive" / "项目报告3.md",
        ROOT / "docs" / "reports" / "archive" / "项目报告4_VC-MCTS预留规划替代RMA版(2).md",
        ROOT / "docs" / "reports" / "archive" / "项目报告4_VC-MCTS预留规划替代RMA版_重构版.md",
        ROOT / "docs" / "reports" / "archive" / "项目报告5_当前代码逻辑重构版.md",
        ROOT / "docs" / "reports" / "archive" / "项目报告6_最终版.md",
        ROOT / "docs" / "reports" / "archive" / "项目报告_完善版.md",
        ROOT / "docs" / "reports" / "archive" / "项目报告_算法解释.md",
        ROOT / "docs" / "reports" / "archive" / "项目阶段性汇报.md",
        ROOT / "docs" / "reports" / "archive" / "项目阶段性汇报_.md",
        ROOT / "docs" / "reports" / "archive" / "项目阶段性汇报_完整版.md",
        ROOT / "references" / "literature",
        ROOT / "legacy" / "MAMHSA_for_fjsp-master",
    ]

    for path in expected_paths:
        assert path.exists(), f"Expected classified path to exist: {path}"

    loose_root_files = {
        "报告.md",
        "项目方案.md",
        "项目建模说明.md",
        "项目报告.md",
        "项目报告3.md",
        "项目报告4_VC-MCTS预留规划替代RMA版(2).md",
        "项目报告4_VC-MCTS预留规划替代RMA版_重构版.md",
        "项目报告5_当前代码逻辑重构版.md",
        "项目报告_完善版.md",
        "项目报告_算法解释.md",
        "项目阶段性汇报.md",
        "项目阶段性汇报_.md",
        "项目阶段性汇报_完整版.md",
        "FAB项目.docx",
        "FAB厂机台组调度问题说明.pptx",
    }
    for filename in loose_root_files:
        assert not (ROOT / filename).exists(), f"Root file should be classified: {filename}"

    assert not (ROOT / "lunwen").exists()
    assert not (ROOT / "MAMHSA_for_fjsp-master").exists()


def test_fabenv_peripheral_files_are_classified():
    expected_paths = [
        FABENV / "scripts" / "run" / "run_phase1_environment_demo.py",
        FABENV / "scripts" / "run" / "run_phase2_sas_inference_demo.py",
        FABENV / "scripts" / "run" / "train_phase2_sas_ppo.py",
        FABENV / "scripts" / "run" / "train_late_hi.py",
        FABENV / "scripts" / "evaluation" / "evaluate_baselines.py",
        FABENV / "scripts" / "evaluation" / "compile_comparison_table.py",
        FABENV / "scripts" / "experiments" / "exp_arrival_prob.py",
        FABENV / "scripts" / "experiments" / "exp_qtime_chain.py",
        FABENV / "scripts" / "experiments" / "tune_arrival_gap.py",
        FABENV / "scripts" / "probes" / "oracle_reservation_probe.py",
        FABENV / "scripts" / "probes" / "probe_topk.py",
        FABENV / "scripts" / "probes" / "vc_mcts_probe.py",
        FABENV / "scripts" / "probes" / "vc_mcts_trace_summary.py",
        FABENV / "notebooks" / "training_visualization.ipynb",
        FABENV / "docs" / "reports" / "报告方案.md",
        FABENV / "artifacts" / "checkpoints",
        FABENV / "artifacts" / "results",
        FABENV / "artifacts" / "pressure_outputs",
        FABENV / "artifacts" / "profiles",
    ]

    for path in expected_paths:
        assert path.exists(), f"Expected FABenv classified path to exist: {path}"

    loose_patterns = ["*.pt", "*.prof", "*.ipynb"]
    for pattern in loose_patterns:
        loose_files = list(FABENV.glob(pattern))
        assert loose_files == [], f"FABenv root has loose generated files: {loose_files}"

    loose_scripts = [
        "compile_comparison_table.py",
        "evaluate_baselines.py",
        "exp_arrival_prob.py",
        "exp_qtime_chain.py",
        "oracle_reservation_probe.py",
        "probe_topk.py",
        "run_phase1_environment_demo.py",
        "run_phase2_sas_inference_demo.py",
        "train_late_hi.py",
        "train_phase2_sas_ppo.py",
        "tune_arrival_gap.py",
        "vc_mcts_probe.py",
        "vc_mcts_trace_summary.py",
        "报告方案.md",
    ]
    for filename in loose_scripts:
        assert not (FABENV / filename).exists(), f"FABenv root file should be classified: {filename}"


def test_moved_fabenv_scripts_bootstrap_bare_imports():
    script_paths = [
        FABENV / "scripts" / "run" / "run_phase1_environment_demo.py",
        FABENV / "scripts" / "run" / "run_phase2_sas_inference_demo.py",
        FABENV / "scripts" / "run" / "train_phase2_sas_ppo.py",
        FABENV / "scripts" / "run" / "train_late_hi.py",
        FABENV / "scripts" / "evaluation" / "evaluate_baselines.py",
        FABENV / "scripts" / "evaluation" / "compile_comparison_table.py",
        FABENV / "scripts" / "experiments" / "exp_arrival_prob.py",
        FABENV / "scripts" / "experiments" / "exp_qtime_chain.py",
        FABENV / "scripts" / "experiments" / "tune_arrival_gap.py",
        FABENV / "scripts" / "probes" / "oracle_reservation_probe.py",
        FABENV / "scripts" / "probes" / "probe_topk.py",
        FABENV / "scripts" / "probes" / "vc_mcts_probe.py",
        FABENV / "scripts" / "probes" / "vc_mcts_trace_summary.py",
    ]

    for script_path in script_paths:
        text = script_path.read_text(encoding="utf-8")
        assert "FABENV_ROOT = Path(__file__).resolve().parents[2]" in text
        assert 'SCRIPT_ROOT = FABENV_ROOT / "scripts"' in text
        assert "sys.path.insert(0, path_text)" in text
        assert 'SCRIPT_ROOT / "evaluation"' in text
        assert 'SCRIPT_ROOT / "probes"' in text
