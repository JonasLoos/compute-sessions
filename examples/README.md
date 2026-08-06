# Example configs

Known-good `~/.config/compute-sessions/config.toml` contents for specific compute sources. Copy the relevant `[sources.*]` table into your config (or the whole file if it's your only source) and adjust the ssh alias.

- [config.toml](config.toml) — fully annotated example covering all three source types (slurm, docker, vast) and the top-level options.
- [hydra.toml](hydra.toml) — the TU Berlin ML-group hydra SLURM cluster.

If you set up a source others might use too, consider PRing its config here.
