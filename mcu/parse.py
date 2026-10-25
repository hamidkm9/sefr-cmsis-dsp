"""Turn results/sizes.txt and results/qemu_runs.txt into the code-size and instruction tables."""
import collections

HARNESS = {"main", "put_u", "rnd", "make_data", "calibrate", "tick_start", "tick_stop", "Reset_Handler",
           "Default_Handler", "semihost", "sh_puts", "sh_exit", "_sbrk", "vectors", "memset", "cfg.0"}

sizes, cur = collections.defaultdict(int), None
for line in open("../results/sizes.txt"):
    if line.startswith("##"):
        cur = line.split("/")[-1].strip()[:-4]
        continue
    size, name = line.split()
    if name not in HARNESS:
        sizes[cur] += int(size)
print("Code size, bytes of .text for everything reachable from the measured functions (CMSIS-DSP and libm included)")
for v in ["sefr", "lda", "gnb"]:
    print(f"  {v:5s} M4F {sizes['m4_' + v]:6d}   M55 {sizes['m55_' + v]:6d}")

print("\nRetired instructions (QEMU -icount shift=0, SysTick calibrated with a 400,000-instruction loop)")
for block in open("../results/qemu_runs.txt").read().split("## ")[1:]:
    head, *lines = block.strip().split("\n")
    kv = [l.split("=") for l in lines]
    per_tick = 400000 / int(kv[0][1])
    cfg, n = [], 1
    for k, v in kv[1:]:
        if k.startswith("cfg_"):
            cfg.append(f"{k[4:]}={v}")
            if k == "cfg_N":
                n = int(v)
            if k == "cfg_K":
                print(f"  {head:8s} {' '.join(cfg)}")
                cfg = []
        elif k.endswith("_all"):
            print(f"      {k[:-4]:18s} {int(v) * per_tick / n:12,.0f} per sample")
        elif k != "lda_status_is_zero":
            print(f"      {k:18s} {int(v) * per_tick:12,.0f}")
        elif v != "1":
            print("      LDA FAILED")
