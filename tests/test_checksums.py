from __future__ import annotations

from betboard_soccer_extension.storage.checksums import sha256_file


def test_sha256_file(tmp_path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("betboard\n")
    assert sha256_file(path) == "0d176e2c8b9043913d65c7d305ea623ae81a4a999adc15a13dea75392ffa1592"
