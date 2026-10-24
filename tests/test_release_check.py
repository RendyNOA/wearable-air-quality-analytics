import json

from scripts.check_public_release import scan


def _clean_repo(root):
    (root / "data" / "demo").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "README.md").write_text("# demo\nMentions an API key and a password in prose only.\n")
    (root / "data" / "demo" / "x.csv").write_text("a,b\n1,2\n")
    (root / "scripts" / "ok.py").write_text("print('hello')\n")


def test_clean_repo_passes(tmp_path):
    _clean_repo(tmp_path)
    assert scan(tmp_path) == []


def test_flags_local_data_private_notes_and_secrets(tmp_path):
    _clean_repo(tmp_path)
    (tmp_path / "notes_PRIVATE.md").write_text("private notes")
    (tmp_path / "data" / "raw").mkdir()
    (tmp_path / "data" / "raw" / "measurements.csv").write_text("x\n1\n")
    (tmp_path / "scripts" / "upload.py").write_text('API_KEY = "ABCD1234EFGH5678"\n')
    found = "\n".join(scan(tmp_path))
    assert "notes_PRIVATE.md" in found
    assert "data/raw/measurements.csv" in found
    assert "API/write key value" in found


def test_flags_notebook_outputs(tmp_path):
    _clean_repo(tmp_path)
    (tmp_path / "notebooks").mkdir()
    nb = {"cells": [{"cell_type": "code", "source": "1+1", "outputs": [{"text": "2"}]}]}
    (tmp_path / "notebooks" / "a.ipynb").write_text(json.dumps(nb))
    assert any("saved outputs" in p for p in scan(tmp_path))
