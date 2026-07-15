from __future__ import annotations

import stat
from pathlib import Path

import pytest

from ai_clinician.cli import _write_json, _write_jsonl, main


@pytest.mark.parametrize("writer", [_write_json, _write_jsonl])
def test_derived_json_outputs_are_owner_only(tmp_path: Path, writer) -> None:
    output = tmp_path / "derived" / "artifact.json"
    output.parent.mkdir()
    output.write_text("existing", encoding="utf-8")
    output.chmod(0o644)

    value = {"synthetic": True} if writer is _write_json else [{"synthetic": True}]
    writer(output, value)

    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_invalid_arguments_do_not_echo_rejected_values(capsys) -> None:
    identifying_value = "王" + "小明"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "timeline",
                "build",
                "--input",
                "synthetic.jsonl",
                "--output",
                "synthetic-output.jsonl",
                "--patient-name",
                identifying_value,
            ]
        )
    assert error.value.code == 2
    assert identifying_value not in capsys.readouterr().err
