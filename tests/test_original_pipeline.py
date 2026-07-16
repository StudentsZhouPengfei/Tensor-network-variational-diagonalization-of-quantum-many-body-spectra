import json
from pathlib import Path

from tnvd.quickstart import run_quickstart


def test_original_engine_quickstart(tmp_path: Path) -> None:
    output = run_quickstart(tmp_path, num_spins=3, epochs=1, max_layers=0, seed=3)
    expected = (
        "ising_mpo.pth",
        "quickstart_config.json",
        "automation_record.log",
        "checkpoints/Physical_tensorsNL=0.pth",
        "checkpoints/Eigen_MPSsNL=0.pth",
        "checkpoints/Minimize_Energy_NL_0.pdf",
    )
    for relative_path in expected:
        assert (output / relative_path).is_file()
    metadata = json.loads((output / "quickstart_config.json").read_text(encoding="utf-8"))
    assert metadata["purpose"] == "small smoke test of the original TNVD execution path"
    assert metadata["config"]["init_strategy"]["mode"] == "random"
