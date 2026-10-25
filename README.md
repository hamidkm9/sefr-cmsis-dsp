# SEFR for CMSIS-DSP: prototype and benchmark

This repo has an f32 prototype of [SEFR](https://arxiv.org/abs/2006.04620) written in the CMSIS-DSP style, plus everything needed to reproduce the numbers in the CMSIS-DSP feature request. SEFR is a linear classifier that trains on the device in one pass, with no gradients, epochs or learning rate.

SEFR collects per-class sums, class counts and per-feature min/max in one pass, then computes the weights and bias from a closed formula. An optional second pass fits LDA on the K class scores and folds it back into the same K x d weights, so prediction is still one matrix-vector product. The code uses only existing CMSIS-DSP functions (`arm_add_f32`, `arm_mat_vec_mult_f32`, `arm_max_f32`, `arm_mat_cholesky_f32`, the triangular solvers) plus `logf`.

The baseline is the classifier CMSIS-DSP already ships: `arm_gaussian_naive_bayes_predict_f32`, with a one-pass on-device trainer (`bench/gnb_train.c`) that uses scikit-learn's formula.

## API

```c
#include "arm_sefr.h"

static float32_t W[K * D], B[K];
static float32_t train_scratch[ARM_SEFR_TRAIN_SCRATCH(D, K)], lda_scratch[ARM_SEFR_LDA_SCRATCH(K)];
arm_sefr_instance_f32 S;

arm_sefr_init_f32(&S, D, K, W, B);
arm_sefr_train_f32(&S, X, labels, N, train_scratch);   /* one pass */
arm_sefr_lda_f32(&S, X, labels, N, lda_scratch);       /* optional second pass */
uint32_t cls = arm_sefr_predict_f32(&S, x, scores);
```

`X` is N x D row major and `labels` holds class indices in [0, K). Each class needs at least one sample, and at least one sample must be outside it.

## Results

### Accuracy on OpenML-CC18 (71 datasets, 5-fold CV, C float32 code)

| | Binary (35 datasets) | Multiclass (36 datasets) |
|---|---|---|
| Gaussian NB (CMSIS predictor) | 73.6 | 66.7 |
| SEFR | 78.2 (better on 21, worse on 14) | 65.6 (better on 19, worse on 17) |
| SEFR + LDA pass | **81.2** (better on 27, worse on 8) | **75.1** (better on 30, worse on 5) |

The C code gives the same predictions as the float64 Python reference (`python/sefr_ref.py`) on 100% of test samples for SEFR and on 99.7% for SEFR + LDA. The one float32 loss is `letter` (26 classes), which drops from 70.2 to 66.3 because its score covariance needs a larger ridge in float32.

### Code size

Bytes of `.text` with `-Os` and `--gc-sections`, counting everything the measured functions pull in (CMSIS-DSP and libm included):

| | Cortex-M4F | Cortex-M55 (Helium) |
|---|---|---|
| SEFR train + predict | 1,482 | 1,876 |
| SEFR + LDA train + predict | 3,910 | 5,458 |
| GNB train + CMSIS predict | 1,432 | 2,256 |

### RAM, in floats

| | Model | Training scratch |
|---|---|---|
| SEFR | K(d + 1) | 2(d + K) |
| SEFR + LDA second pass | same model | 5K² + 2K |
| GNB | 2Kd + K + 1 | none |

For example, with d = 64 and K = 10, the SEFR model is 2.6 KB and training needs 0.6 KB of scratch, plus 2.1 KB for the LDA pass. The GNB model is 5.2 KB.

### Instructions

These are retired instructions measured with QEMU 8.2 `-icount shift=0` on `mps2-an386` (Cortex-M4F) and `mps3-an547` (Cortex-M55), with SysTick calibrated against a loop of known length. They are not cycles. On a Cortex-M4F most FP and ALU instructions take one cycle and loads take two, so expect roughly 1.2 to 2x as many cycles. The data is synthetic.

| N, d, K | | SEFR train | LDA pass | Predict per sample | GNB train | GNB predict per sample |
|---|---|---|---|---|---|---|
| 150, 4, 3 | M4F | 21k | 112k | 272 | 14k | 1,545 |
| | M55 | 18k | 89k | 251 | 14k | 719 |
| 500, 16, 4 | M4F | 208k | 506k | 437 | 162k | 6,602 |
| | M55 | 174k | 370k | 364 | 147k | 1,977 |
| 1000, 64, 10 | M4F | 1,533k | 5,026k | 3,016 | 1,246k | 63,229 |
| | M55 | 1,272k | 2,932k | 1,661 | 1,119k | 17,138 |

The CMSIS GNB predictor computes `log(2 pi sigma)` for every class and feature on every call, and that accounts for most of its cost. A GNB with those terms precomputed would be much closer to a linear predict. So the prediction gap here isn't a fair measure of SEFR against naive Bayes in general.

## Reproduce

```sh
git clone --recurse-submodules https://github.com/hamidkm9/sefr-cmsis-dsp
cd sefr-cmsis-dsp
```

CMSIS-DSP (v1.18.0-4-g83a2d7bc) and CMSIS_6 (v6.4.0-2-gacf7392d) are pinned as submodules in `third_party/`.

**Accuracy (host, macOS or Linux):**

```sh
python3 -m venv .venv && .venv/bin/pip install -r python/requirements.txt
.venv/bin/python python/prep_cc18.py          # downloads CC18 into data/ (pass OpenML ids to fetch only some)
cmake -S host -B host/build -DCMAKE_BUILD_TYPE=Release && cmake --build host/build
.venv/bin/python python/check_cc18.py         # writes results/cc18_c.jsonl
.venv/bin/python python/report.py
```

**Code size and instructions (Docker; Arm GCC 13.2 and QEMU 8.2 from Ubuntu 24.04):**

```sh
docker build -t sefr-mcu mcu
docker run --rm -v "$PWD:/w" -w /w/mcu sefr-mcu sh run.sh   # writes results/sizes.txt, results/qemu_runs.txt
```

A fresh clone gives instruction counts identical to the ones in `results/`.

## Layout

```
src/        arm_sefr.h, arm_sefr_f32.c   the prototype
bench/      gnb_train.c                  on-device GNB trainer for the baseline
host/       CMake build of a shared library for the Python check
python/     data prep, float64 reference, C-vs-reference check, report
mcu/        bare-metal harness, linker script, Makefile, Dockerfile, QEMU runner
results/    raw outputs behind the tables above
```

## License

Apache-2.0, the same as CMSIS-DSP.
