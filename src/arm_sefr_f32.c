/*
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Hamidreza Keshavarz
 */
#include <math.h>
#include "arm_sefr.h"
#include "dsp/basic_math_functions.h"
#include "dsp/fast_math_functions.h"
#include "dsp/matrix_functions.h"
#include "dsp/statistics_functions.h"
#include "dsp/support_functions.h"

void arm_sefr_init_f32(arm_sefr_instance_f32 *S, uint32_t vectorDimension, uint32_t numberOfClasses,
                       float32_t *pWeights, float32_t *pBias)
{
  S->vectorDimension = vectorDimension;
  S->numberOfClasses = numberOfClasses;
  S->pWeights = pWeights;
  S->pBias = pBias;
}

arm_status arm_sefr_train_f32(arm_sefr_instance_f32 *S, const float32_t *pSrc, const uint32_t *pLabels,
                              uint32_t numSamples, float32_t *pScratch)
{
  const uint32_t d = S->vectorDimension, K = S->numberOfClasses;
  float32_t *W = S->pWeights, *b = S->pBias;
  float32_t *lo = pScratch, *hi = pScratch + d, *n = pScratch + 2U * d, *sn = pScratch + 2U * d + K;
  const float32_t N = (float32_t)numSamples;
  uint32_t i, j, k;

  /* One pass: class sums go straight into the weight buffer. */
  arm_fill_f32(0.0f, W, K * d);
  arm_fill_f32(0.0f, n, K);
  arm_copy_f32(pSrc, lo, d);
  arm_copy_f32(pSrc, hi, d);
  for (i = 0; i < numSamples; i++)
  {
    const float32_t *x = pSrc + i * d;
    k = pLabels[i];
    if (k >= K)
    {
      return ARM_MATH_ARGUMENT_ERROR;
    }
    arm_add_f32(W + k * d, x, W + k * d, d);
    n[k] += 1.0f;
    for (j = 0; j < d; j++)
    {
      lo[j] = x[j] < lo[j] ? x[j] : lo[j];
      hi[j] = x[j] > hi[j] ? x[j] : hi[j];
    }
  }
  for (k = 0; k < K; k++)
  {
    if (n[k] < 1.0f || n[k] > N - 1.0f)
    {
      return ARM_MATH_ARGUMENT_ERROR;
    }
  }

  /* Per feature: mean of the class (p) and of the rest (r) in min-max scaled space, w = (p - r) / (p + r).
     b and sn accumulate w . p and w . r for the threshold. */
  arm_fill_f32(0.0f, b, K);
  arm_fill_f32(0.0f, sn, K);
  for (j = 0; j < d; j++)
  {
    float32_t total = 0.0f;
    for (k = 0; k < K; k++)
    {
      total += W[k * d + j];
    }
    hi[j] = hi[j] - lo[j] > 0.0f ? hi[j] - lo[j] : 1.0f; /* hi now holds the range */
    for (k = 0; k < K; k++)
    {
      const float32_t p = (W[k * d + j] / n[k] - lo[j]) / hi[j];
      const float32_t r = ((total - W[k * d + j]) / (N - n[k]) - lo[j]) / hi[j];
      const float32_t w = (p - r) / (p + r + 1e-7f);
      W[k * d + j] = w;
      b[k] += w * p;
      sn[k] += w * r;
    }
  }

  /* Threshold t = ((N - n) w.p + n w.r) / N, weighted by class counts; then fold the scaling into W, b. */
  for (k = 0; k < K; k++)
  {
    float32_t *w = W + k * d;
    float32_t wlo = 0.0f;
    const float32_t t = ((N - n[k]) * b[k] + n[k] * sn[k]) / N;
    for (j = 0; j < d; j++)
    {
      w[j] /= hi[j];
      wlo += w[j] * lo[j];
    }
    b[k] = -(t + wlo);
  }
  return ARM_MATH_SUCCESS;
}

arm_status arm_sefr_lda_f32(arm_sefr_instance_f32 *S, const float32_t *pSrc, const uint32_t *pLabels,
                            uint32_t numSamples, float32_t *pScratch)
{
  const uint32_t d = S->vectorDimension, K = S->numberOfClasses;
  float32_t *M = pScratch;            /* K x K class means of the scores */
  float32_t *C = M + K * K;           /* K x K within-class scatter, later the solution A^T */
  float32_t *L = C + K * K;           /* Cholesky factor */
  float32_t *Lt = L + K * K;          /* its transpose */
  float32_t *Y = Lt + K * K;          /* intermediate solve */
  float32_t *n = Y + K * K;           /* class counts */
  float32_t *z = n + K;               /* scores of one sample, then one weight column */
  arm_matrix_instance_f32 mM, mC, mL, mLt, mY;
  const float32_t N = (float32_t)numSamples;
  uint32_t i, j, k, l;
  arm_status status;

  /* One pass (Welford): running class means of the scores and the pooled within-class scatter. */
  arm_fill_f32(0.0f, M, 2U * K * K);
  arm_fill_f32(0.0f, n, K);
  for (i = 0; i < numSamples; i++)
  {
    float32_t *m;
    k = pLabels[i];
    m = M + k * K;
    arm_sefr_predict_f32(S, pSrc + i * d, z);
    n[k] += 1.0f;
    /* L, Lt and Y are free until the solve, so they hold per-sample temporaries here. */
    arm_sub_f32(z, m, L, K);                   /* z - old mean */
    arm_scale_f32(L, 1.0f / n[k], Lt, K);
    arm_add_f32(m, Lt, m, K);                  /* new mean */
    arm_sub_f32(z, m, Lt, K);                  /* z - new mean */
    for (l = 0; l < K; l++)
    {
      arm_scale_f32(Lt, L[l], Y, K);
      arm_add_f32(C + l * K, Y, C + l * K, K);
    }
  }
  float32_t trace = 0.0f;
  arm_scale_f32(C, 1.0f / N, C, K * K);
  for (k = 0; k < K; k++)
  {
    trace += C[k * K + k];
  }
  /* Solve C A^T = M^T with C = L L^T. The ridge starts at 1e-6 of the mean variance, as in the float64
     reference; float32 sometimes needs more when the K scores are nearly collinear, so it grows 10x per retry. */
  arm_mat_init_f32(&mC, K, K, C);
  arm_mat_init_f32(&mL, K, K, L);
  arm_mat_init_f32(&mLt, K, K, Lt);
  arm_mat_init_f32(&mY, K, K, Y);
  arm_mat_init_f32(&mM, K, K, M);
  float32_t ridge = 1e-6f * trace / (float32_t)K + 1e-30f, added = 0.0f;
  for (l = 0; l < 6; l++, ridge *= 10.0f)
  {
    for (k = 0; k < K; k++)
    {
      C[k * K + k] += ridge - added;
    }
    added = ridge;
    status = arm_mat_cholesky_f32(&mC, &mL);
    if (status == ARM_MATH_SUCCESS)
    {
      break;
    }
  }
  if (status != ARM_MATH_SUCCESS)
  {
    return status;
  }
  arm_mat_trans_f32(&mL, &mLt);
  arm_mat_trans_f32(&mM, &mC);        /* C now holds M^T */
  status = arm_mat_solve_lower_triangular_f32(&mL, &mC, &mY);
  if (status != ARM_MATH_SUCCESS)
  {
    return status;
  }
  status = arm_mat_solve_upper_triangular_f32(&mLt, &mY, &mC);  /* C = A^T */
  if (status != ARM_MATH_SUCCESS)
  {
    return status;
  }

  /* New bias: log prior + A b - 0.5 a_k . m_k, using the old b before it is overwritten. */
  for (k = 0; k < K; k++)
  {
    float32_t ab = 0.0f, am = 0.0f;
    for (l = 0; l < K; l++)
    {
      ab += C[l * K + k] * S->pBias[l];
      am += C[l * K + k] * M[k * K + l];
    }
    z[k] = logf(n[k] / N) + ab - 0.5f * am;
  }
  arm_copy_f32(z, S->pBias, K);

  /* W <- A W, one weight column at a time, so no K x d scratch is needed. */
  for (j = 0; j < d; j++)
  {
    for (k = 0; k < K; k++)
    {
      float32_t s = 0.0f;
      for (l = 0; l < K; l++)
      {
        s += C[l * K + k] * S->pWeights[l * d + j];
      }
      z[k] = s;
    }
    for (k = 0; k < K; k++)
    {
      S->pWeights[k * d + j] = z[k];
    }
  }
  return ARM_MATH_SUCCESS;
}

uint32_t arm_sefr_predict_f32(const arm_sefr_instance_f32 *S, const float32_t *pIn, float32_t *pScores)
{
  arm_matrix_instance_f32 mW;
  float32_t best;
  uint32_t index;
  arm_mat_init_f32(&mW, S->numberOfClasses, S->vectorDimension, S->pWeights);
  arm_mat_vec_mult_f32(&mW, pIn, pScores);
  arm_add_f32(pScores, S->pBias, pScores, S->numberOfClasses);
  arm_max_f32(pScores, S->numberOfClasses, &best, &index);
  return index;
}
