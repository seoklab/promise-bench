"""Adds ``-C / --data-root`` option to any click command.

Usage
-----
Change ``@click.command(...)`` to ``@click.command(cls=DataRootCommand, ...)``.

When the user passes ``-C data-90``, every click option/argument whose
**default** starts with ``data/`` (or equals ``data``) is automatically
rebased so that ``data/`` is replaced by ``data-90/``.

The ``-C`` value itself is consumed internally and is **not** forwarded to
the decorated function.
"""

from __future__ import annotations

from pathlib import Path

import click


def _rebase(val, data_root: Path):
    """Replace a ``data/…`` prefix with *data_root*."""
    if isinstance(val, Path):
        try:
            rel = val.relative_to("data")
            return data_root / rel
        except ValueError:
            return val
    elif isinstance(val, str):
        if val == "data":
            return str(data_root)
        if val.startswith("data/") or val.startswith("data\\"):
            return str(data_root / val[5:])
    return val


class DataRootCommand(click.Command):
    """``click.Command`` subclass that injects a ``-C / --data-root`` option.

    Before click validates defaults (e.g. ``exists=True``), any param whose
    default starts with ``data/`` is rebased to the user-supplied root.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.insert(
            0,
            click.Option(
                ["-C", "--data-root"],
                type=click.Path(path_type=Path),
                default=None,
                help="Root data directory; rebases all data/ defaults.",
            ),
        )

    # ------------------------------------------------------------------

    def make_context(self, info_name, args, parent=None, **extra):
        data_root = self._find_data_root(args)

        saved: dict[str, object] = {}
        if data_root is not None:
            for param in self.params:
                if param.name == "data_root":
                    continue
                if param.default is not None:
                    saved[param.name] = param.default
                    param.default = _rebase(param.default, data_root)

        try:
            ctx = super().make_context(info_name, args, parent=parent, **extra)
        finally:
            # Restore originals so the Command object stays reusable.
            for param in self.params:
                if param.name in saved:
                    param.default = saved[param.name]

        # Don't forward data_root to the decorated function.
        ctx.params.pop("data_root", None)
        return ctx

    # ------------------------------------------------------------------

    @staticmethod
    def _find_data_root(args):
        for i, arg in enumerate(args):
            if arg in ("-C", "--data-root") and i + 1 < len(args):
                return Path(args[i + 1])
            if arg.startswith("--data-root="):
                return Path(arg.split("=", 1)[1])
        return None
