# HyperspectralFusionMetrics

HyperspectralFusionMetrics is a Python tool for hyperspectral image fusion
quality assessment using PSNR, SSIM, SAM, ERGAS, RMSE, and CC.

It computes:

- PSNR
- SSIM
- SAM
- ERGAS
- RMSE
- CC

## Project structure

```text
HyperspectralFusionMetrics
├─ DATA
│  ├─ FUSED
│  ├─ REF
│  └─ OUTPUT
├─ METRIC
│  ├─ compute_metrics.py
│  └─ README.md
├─ LICENSE
├─ README.md
└─ requirements.txt
```

Put fused `.mat` files in:

```text
DATA/FUSED
```

Put the ground-truth/reference `.mat` file in:

```text
DATA/REF
```

Metric results are written to:

```text
DATA/OUTPUT
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run from the project root:

```bash
python METRIC/compute_metrics.py
```

The default ERGAS ratio is `12`. To use another ratio:

```bash
python METRIC/compute_metrics.py --ratio 3
```

## Data layout

The script uses `(rows, cols, bands)` internally. It automatically converts
common hyperspectral layouts such as `(bands, rows, cols)` when the converted
shape matches the reference image.

After shape conversion, HyperspectralFusionMetrics also checks two spatial orientations:

```text
normal:  rows x cols x bands
swap_hw: cols x rows x bands
```

If both orientations match the reference shape, HyperspectralFusionMetrics computes metrics for
both and keeps the orientation with the higher PSNR. The selected orientation
is written as `ORIENTATION` in the output files.

For example:

```text
FUSED: (54, 240, 240)
REF:   (240, 240, 54)
```

The fused image is converted to:

```text
(240, 240, 54)
```

## Output

For each fused file, HyperspectralFusionMetrics writes:

```text
DATA/OUTPUT/<name>_metrics.txt
DATA/OUTPUT/<name>_metrics.json
DATA/OUTPUT/<name>_psnr_per_band.csv
DATA/OUTPUT/metrics_summary.csv
```

## License

Code is released under the MIT License. See [LICENSE](LICENSE).
