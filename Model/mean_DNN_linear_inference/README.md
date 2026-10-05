# Protein localization example

Use this example to predict protein localization from 1,280 mean ESM2 features
with the bundled 10-class linear classifier. From the repository root:

```bash
python Model/mean_DNN_linear_inference/code/mean_DNN_linear_mean_inference.py \
  --features Model/mean_DNN_linear_inference/output/sample_data_feature_rep.xlsx \
  --output-dir results/mean-example
```

This route requires PyTorch, pandas, NumPy and openpyxl. It reuses 270 bundled
feature rows and the small project-trained classifier; no ESM2 download is needed.
Input columns must be `Entry`, `Sequence`, then `ESM2_mean0` through
`ESM2_mean1279` in order, with finite numeric values. Output is
`results/mean-example/sample_data_prediction.xlsx`, with `Entry`,
`predict_topic` and `predict_probability`. Compare IDs and labels against
`Model/mean_DNN_linear_inference/output/sample_data_prediction.xlsx`; maximum
class probabilities should agree within 1e-5. The prior local 270-row classifier
check took 9.21 seconds including imports and comparison. The entry point using
this GitHub layout was also checked against the reference; timing depends on
hardware and the installed environment.

## Raw sequences

For Excel input, supply `Entry` and `Sequence` columns explicitly:

```bash
python Model/mean_DNN_linear_inference/code/mean_DNN_linear_mean_inference.py \
  --input Model/mean_DNN_linear_inference/data/in/Excel_format_protein_sequence_data_example.xlsx \
  --output-dir results/raw-mean
```

Raw extraction additionally needs Transformers and protloc-mex-x and the full
`facebook/esm2_t33_650M_UR50D` model. Choose `download` at the prompt, or `local`
when `data/local_model/` contains the complete pretrained model. The small
project classifiers are separate from the ESM2 backbone. Full extraction and
segment0 prediction were not executed; their runtime is not measured.

The alternative script is `code/mean_DNN_linear_segment0_mean_inference.py`;
pass the same explicit `--input` and a separate `--output-dir`. Raw extraction
writes `<stem>_feature_rep.xlsx` and `<stem>_prediction.xlsx`.

For FASTA/FAA, run `python Model/mean_DNN_linear_inference/code/data_prepared.py`.
It converts files from `data/in/` to `data/out/` under this example directory;
pass the converted workbook with `--input`. Conversion overwrites output files
with the same stem. Keep separate output directories for different runs.
