/*
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Hamidreza Keshavarz
 */
/* On-device Gaussian naive Bayes training (one pass, sklearn's formula) for the CMSIS predictor. */
#include "arm_math_types.h"
#include "dsp/bayes_functions.h"
#include "dsp/support_functions.h"

void gnb_train_f32(const float32_t *X, const uint32_t *y, uint32_t N, uint32_t d, uint32_t K,
                   float32_t *theta, float32_t *sigma, float32_t *priors, float32_t *epsilon)
{
  uint32_t i, j, k;
  arm_fill_f32(0.0f, theta, K * d);
  arm_fill_f32(0.0f, sigma, K * d);
  arm_fill_f32(0.0f, priors, K);
  for (i = 0; i < N; i++)
  {
    const float32_t *x = X + i * d;
    k = y[i];
    priors[k] += 1.0f;
    for (j = 0; j < d; j++)
    {
      theta[k * d + j] += x[j];
      sigma[k * d + j] += x[j] * x[j];
    }
  }
  float32_t vmax = 0.0f;
  for (k = 0; k < K; k++)
  {
    for (j = 0; j < d; j++)
    {
      const float32_t m = theta[k * d + j] / priors[k];
      float32_t v = sigma[k * d + j] / priors[k] - m * m;
      v = v > 0.0f ? v : 0.0f;
      theta[k * d + j] = m;
      sigma[k * d + j] = v;
    }
    priors[k] /= (float32_t)N;
  }
  /* sklearn: var_smoothing (1e-9) times the largest variance of the whole data set */
  for (j = 0; j < d; j++)
  {
    float32_t s = 0.0f, ss = 0.0f;
    for (i = 0; i < N; i++)
    {
      s += X[i * d + j];
      ss += X[i * d + j] * X[i * d + j];
    }
    const float32_t v = ss / N - (s / N) * (s / N);
    vmax = v > vmax ? v : vmax;
  }
  *epsilon = 1e-9f * vmax;
}

uint32_t gnb_predict_f32(const float32_t *x, uint32_t d, uint32_t K, const float32_t *theta,
                         const float32_t *sigma, const float32_t *priors, float32_t epsilon, float32_t *buf)
{
  arm_gaussian_naive_bayes_instance_f32 S = {d, K, theta, sigma, priors, epsilon};
  return arm_gaussian_naive_bayes_predict_f32(&S, x, buf, buf + K);
}
