import subprocess
import sys

from pgvb import __version__
from pgvb.__main__ import main


def test_version_string_is_set():
    assert __version__


def test_main_version_command_prints_version(capsys):
    exit_code = main(["version"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == __version__


def test_console_script_installed():
    result = subprocess.run(
        [sys.executable, "-m", "pgvb", "version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == __version__
