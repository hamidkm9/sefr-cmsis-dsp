/* Instruction counts (QEMU -icount, SysTick) and code size for SEFR, SEFR + LDA and Gaussian NB on Cortex-M. */
#include <stdint.h>
#include <math.h>
#include "arm_math_types.h"
#include "arm_sefr.h"
#include "dsp/bayes_functions.h"

void sh_puts(const char *s);
void gnb_train_f32(const float32_t *X, const uint32_t *y, uint32_t N, uint32_t d, uint32_t K,
                   float32_t *theta, float32_t *sigma, float32_t *priors, float32_t *epsilon);

#define SYST_CSR (*(volatile uint32_t *)0xE000E010)
#define SYST_RVR (*(volatile uint32_t *)0xE000E014)
#define SYST_CVR (*(volatile uint32_t *)0xE000E018)

#define NMAX 1000
#define DMAX 64
#define KMAX 10
static float32_t X[NMAX * DMAX];
static uint32_t Y[NMAX];
static float32_t W[KMAX * DMAX], B[KMAX], scratch[5 * KMAX * KMAX + 2 * KMAX + 2 * DMAX];
static float32_t theta[KMAX * DMAX], sigma[KMAX * DMAX], priors[KMAX], buf[2 * KMAX];
volatile uint32_t sink;

static void put_u(const char *label, uint32_t v)
{
  char s[64], t[12];
  int i = 0, j = 0;
  while (label[i]) { s[i] = label[i]; i++; }
  do { t[j++] = '0' + v % 10; v /= 10; } while (v);
  while (j) s[i++] = t[--j];
  s[i++] = '\n'; s[i] = 0;
  sh_puts(s);
}

static uint32_t lcg = 12345;
static float32_t rnd(void) { lcg = lcg * 1664525u + 1013904223u; return (float32_t)(lcg >> 8) / 16777216.0f; }

static void make_data(uint32_t N, uint32_t d, uint32_t K)
{
  for (uint32_t i = 0; i < N; i++)
  {
    uint32_t k = i % K;
    Y[i] = k;
    for (uint32_t j = 0; j < d; j++) X[i * d + j] = rnd() + 0.5f * (float32_t)((j + k) % K == 0);
  }
}

static void tick_start(void) { SYST_RVR = 0xFFFFFF; SYST_CVR = 0; SYST_CSR = 5; }
static uint32_t tick_stop(void) { uint32_t v = 0xFFFFFF - SYST_CVR; SYST_CSR = 0; return v; }

/* 4 instructions per iteration: subs, add, cmp... measured as a calibration of ticks per instruction. */
static uint32_t calibrate(void)
{
  tick_start();
  __asm__ volatile(
      "mov r0, #0\n ldr r1, =100000\n"
      "1: add r0, r0, #1\n add r2, r2, #0\n cmp r0, r1\n bne 1b\n" ::: "r0", "r1", "r2", "cc");
  return tick_stop(); /* 400000 loop instructions + a few */
}

int main(void)
{
  arm_sefr_instance_f32 S;
  static const uint32_t cfg[][3] = {{150, 4, 3}, {500, 16, 4}, {1000, 64, 10}};
  put_u("calib_ticks_per_400000_insn=", calibrate());
  for (uint32_t c = 0; c < 3; c++)
  {
    const uint32_t N = cfg[c][0], d = cfg[c][1], K = cfg[c][2];
    uint32_t t, i;
    float32_t eps;
    make_data(N, d, K);
    put_u("cfg_N=", N); put_u("cfg_d=", d); put_u("cfg_K=", K);
#if defined(WITH_SEFR)
    arm_sefr_init_f32(&S, d, K, W, B);
    tick_start(); sink = arm_sefr_train_f32(&S, X, Y, N, scratch); t = tick_stop();
    put_u("sefr_train=", t);
    tick_start(); for (i = 0; i < N; i++) sink += arm_sefr_predict_f32(&S, X + i * d, buf); t = tick_stop();
    put_u("sefr_predict_all=", t);
#endif
#if defined(WITH_LDA)
    tick_start(); sink = arm_sefr_lda_f32(&S, X, Y, N, scratch); t = tick_stop();
    put_u("sefr_lda_pass=", t);
    put_u("lda_status_is_zero=", sink == 0);
#endif
#if defined(WITH_GNB)
    tick_start(); gnb_train_f32(X, Y, N, d, K, theta, sigma, priors, &eps); t = tick_stop();
    put_u("gnb_train=", t);
    arm_gaussian_naive_bayes_instance_f32 G = {d, K, theta, sigma, priors, eps};
    tick_start(); for (i = 0; i < N; i++) sink += arm_gaussian_naive_bayes_predict_f32(&G, X + i * d, buf, buf + K); t = tick_stop();
    put_u("gnb_predict_all=", t);
#endif
  }
  return 0;
}
