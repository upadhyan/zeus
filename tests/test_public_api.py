"""The public surface advertised by `zeus.__init__`."""
import zeus


def test_top_level_exports():
    assert hasattr(zeus, "Zeus")
    assert hasattr(zeus, "ZeusClusterer")
    assert hasattr(zeus, "__version__")
    assert isinstance(zeus.__version__, str)


def test_no_torch_train_imports_on_module_load():
    """Importing zeus must not pull in wandb, openml, or omegaconf."""
    import sys
    for forbidden in ("wandb", "openml", "omegaconf"):
        assert forbidden not in sys.modules, (
            f"`import zeus` should not have imported {forbidden}"
        )
