/*
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Hamidreza Keshavarz
 */
/*
 * SEFR: closed-form on-device training of a linear classifier, written in the CMSIS-DSP style.
 *
 * Training is one pass over the data (per-class sums, class counts, per-feature min/max) and an
 * O(K d) finish. The optional second stage (arm_sefr_lda_f32) is one more pass that fits LDA on the
 * K class scores and folds it back into the same K x d weights, so prediction cost is unchanged.
 */
#ifndef ARM_SEFR_H_
#define ARM_SEFR_H_

#include "arm_math_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
  uint32_t vectorDimension;   /**< Dimension of vector space (d) */
  uint32_t numberOfClasses;   /**< Number of classes (K) */
  float32_t *pWeights;        /**< K x d weights, row major */
  float32_t *pBias;           /**< K biases */
} arm_sefr_instance_f32;

/** Scratch floats needed by arm_sefr_train_f32 */
#define ARM_SEFR_TRAIN_SCRATCH(d, K) (2U * ((d) + (K)))
/** Scratch floats needed by arm_sefr_lda_f32 */
#define ARM_SEFR_LDA_SCRATCH(K) (5U * (K) * (K) + 2U * (K))

void arm_sefr_init_f32(arm_sefr_instance_f32 *S, uint32_t vectorDimension, uint32_t numberOfClasses,
                       float32_t *pWeights, float32_t *pBias);

/**
 * pSrc is numSamples x d, row major. pLabels holds class indices in [0, K).
 * Every class must have at least one sample and at least one sample must be outside it.
 */
arm_status arm_sefr_train_f32(arm_sefr_instance_f32 *S, const float32_t *pSrc, const uint32_t *pLabels,
                              uint32_t numSamples, float32_t *pScratch);

/** Second stage: LDA on the K scores of an already trained model, folded back into its weights. */
arm_status arm_sefr_lda_f32(arm_sefr_instance_f32 *S, const float32_t *pSrc, const uint32_t *pLabels,
                            uint32_t numSamples, float32_t *pScratch);

/** Returns the predicted class; pScores receives the K scores. */
uint32_t arm_sefr_predict_f32(const arm_sefr_instance_f32 *S, const float32_t *pIn, float32_t *pScores);

#ifdef __cplusplus
}
#endif

#endif
