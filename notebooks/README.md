# Notebook Maintenance

Colab notebooks in this directory should stay thin. Shared setup lives in
`scripts/colab_bootstrap.py`; experiment logic should live in `src/` or `scripts/`.

When adding a new Colab notebook:

1. Keep a single settings cell at the top.
2. Use `sync_repo(...)` from `scripts/colab_bootstrap.py` instead of copying git clone code.
3. Use `install_mamba_env(...)` or `install_base_env(...)` instead of copying pip commands.
4. Move reusable data loading, reconstruction, and metric code into package modules.
