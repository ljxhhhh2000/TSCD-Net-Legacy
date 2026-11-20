# TSCD-Net implementation validation

This records the initial validation. Later full experiment results are in [results/main](results/main).

Validation completed on 2026-11-15 in the local Windows environment.

## Static and unit checks

- Shared environment: `E:\Study\Article\.venv-demo`.
- Python 3.10.20, PyTorch 2.5.1+cu121, CUDA 12.1.
- `pip check`: no broken requirements.
- `tscd-net==0.2.0`: installed as an editable local package.
- `matplotlib==3.10.9`: installed for trajectory PNG output.
- `python -m tscd_net --help`: passed.
- `python -m compileall -q src tests`: passed.
- `python -m pytest -q`: 7 tests passed.
- Tests cover the paper's 15-symptom inventory, recent-post window, minimum
  post count after filtering, 70/10/20 split, Time2Vec dimensions, TSCD-Net
  forward pass, and padding masking.
- A CUDA forward/backward smoke check with `batch=32`, `symptoms=15`,
  `timesteps=500`, `d_model=64`, and 2 attention layers completed on the local
  4 GB GTX 1650 Ti. Peak allocation was about 2.0 GB.

## End-to-end SWDD demo

Command:

```powershell
& 'E:\Study\Article\.venv-demo\Scripts\Activate.ps1'
python -m tscd_net_legacy all --config configs/demo.json --force
```

The demo scanned the first 50 JSONL records from each class and extracted:

- 79 eligible users: 39 control and 40 depressed;
- 15 symptom channels;
- sequence length 54 to 459 with a mean of 189.96;
- 15,320 retained posts;
- sentence embeddings from the local multilingual SBERT model.

Training, validation, testing, metric output, symptom alignment, and symptom
trajectory export all completed. Demo metrics are not scientific results.

## Important limitation

The full 19,526-control/3,711-depressed experiment has not been recorded as a
completed result in this directory. The command is ready, but feature
extraction and five full training runs are intentionally left as the formal
experiment rather than being mixed with the smoke-test numbers.
