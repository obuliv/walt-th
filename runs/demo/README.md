# Demo run

Checked-in output of `scripts/build_best_rm.sh` (best-known config: `lr_v6`,
`--ignore-sql-good --drop-bad-vs-bad-pairs`, ollama-only severity dataset, `C=30` —
see `docs/experiments.md`), so reviewers can run evaluation without repeating the
Ollama/Claude data-generation steps.

- `synth_severity_data.jsonl.gz` — stage 1 (Spider extraction + llama3.2 `sql_bad`)
- `synth_severity_enhanced.jsonl.gz` — stage 2 (Claude reason + severity)
- `synth_severity_enhanced_ollamaonly.jsonl.gz` — stage 3 (llama3.2-only `sql_bad`), the eval input
- `rm_model.joblib` / `rm_metrics.json` — stage 5, the trained model

The full C-sweep run log behind the `C=30` choice isn't included here (see
`docs/experiments.md` for that).

Datasets are gzipped (`walt.rm.data.gen_training_data.load_records`/`write_jsonl` and
`walt.rm.model.base.load_examples` read/write `.gz` transparently via
`walt.utils.jsonl_io.open_jsonl` — any `.jsonl` path works uncompressed too).

To run evaluation alone (still needs the Spider SQLite DBs locally at
`$DATA_PATH/spider` and a local Ollama server with `llama3.2` pulled, per
`CLAUDE.md`):

```bash
scripts/evaluate_best.sh --output-dir runs/demo
```
