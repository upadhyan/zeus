"""The public surface advertised by `zeus.__init__`."""
import subprocess
import sys

import zeus


def test_top_level_exports():
    assert hasattr(zeus, "Zeus")
    assert hasattr(zeus, "ZeusClusterer")
    assert hasattr(zeus, "__version__")
    assert isinstance(zeus.__version__, str)


def test_no_torch_train_imports_on_module_load():
    """Importing zeus must not pull in wandb, openml, or omegaconf.

    Runs in a subprocess so sibling test modules that legitimately import
    openml (e.g. test_openml_regression.py) don't taint sys.modules.
    """
    code = (
        "import sys; import zeus; "
        "forbidden = ('wandb', 'openml', 'omegaconf'); "
        "leaked = [m for m in forbidden if m in sys.modules]; "
        "assert not leaked, leaked"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"`import zeus` leaked forbidden modules: {result.stderr}"
    )
