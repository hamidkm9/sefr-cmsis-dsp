#include <stdint.h>
extern uint32_t __bss_start, __bss_end, __stack_top;
int main(void);
void Reset_Handler(void);
static void Default_Handler(void) { for (;;) {} }
__attribute__((section(".vectors"), used)) static void (*const vectors[16])(void) = {
  (void (*)(void))&__stack_top, Reset_Handler, Default_Handler, Default_Handler, Default_Handler,
  Default_Handler, Default_Handler, 0, 0, 0, 0, Default_Handler, Default_Handler, 0, Default_Handler, Default_Handler};

static int semihost(int op, const void *arg)
{
  register int r0 __asm__("r0") = op;
  register const void *r1 __asm__("r1") = arg;
  __asm__ volatile("bkpt 0xab" : "+r"(r0) : "r"(r1) : "memory");
  return r0;
}
void sh_puts(const char *s) { semihost(0x04, s); }
void sh_exit(void) { semihost(0x18, (const void *)0x20026); for (;;) {} }

void Reset_Handler(void)
{
  for (uint32_t *p = &__bss_start; p < &__bss_end; p++) *p = 0;
  *(volatile uint32_t *)0xE000ED88 |= (0xFU << 20); /* CPACR: enable the FPU (and MVE on M55) */
  __asm__ volatile("dsb\n isb");
  main();
  sh_exit();
}
/* newlib stubs: nothing here allocates or does I/O through libc */
void *_sbrk(int n) { (void)n; return (void *)-1; }
