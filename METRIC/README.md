# Metric computation

Install dependencies from the project root:

```powershell
pip install -r requirements.txt
```

Put fused `.mat` files in:

```text
DATA/FUSED
```

Put the ground-truth/reference `.mat` file in:

```text
DATA/REF
```

Run from the project root:

```powershell
python .\METRIC\compute_metrics.py
```

Results are written to:

```text
DATA/OUTPUT
```

The script computes `PSNR`, `SSIM`, `SAM`, `ERGAS`, `RMSE`, and `CC`.

## Metric formulas

![Metric computation formulas](../IMG/image.png)

It automatically converts common hyperspectral layouts such as
`(bands, rows, cols)` to `(rows, cols, bands)`.

It also tests normal H/W orientation and swapped H/W orientation. If both are
valid, the result with higher PSNR is selected and recorded as `ORIENTATION`.

The default ERGAS ratio is `12`. To use another ratio:

```powershell
python .\METRIC\compute_metrics.py --ratio 3
```

## License

MIT. See [LICENSE](../LICENSE).
