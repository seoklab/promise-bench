"""Evaluation modules for ProMiSE-bench."""

__all__ = [
    # esm_run
    "ESMMSAPredictor",
    "read_a3m", 
    "random_sample",
    "greedy_select",
    "process_a3m_file",
    "discover_a3m_files",
    # cif_to_renumbered_pdb
    "cif_to_renumbered_pdb",
    "create_renumber_mapping",
    "parse_a3m",
]

def __getattr__(name):
    if name in ("ESMMSAPredictor", "read_a3m", "random_sample", 
                "greedy_select", "process_a3m_file", "discover_a3m_files"):
        from . import esm_run
        return getattr(esm_run, name)
    elif name in ("cif_to_renumbered_pdb", "create_renumber_mapping", "parse_a3m"):
        from . import cif_to_renumbered_pdb
        return getattr(cif_to_renumbered_pdb, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
