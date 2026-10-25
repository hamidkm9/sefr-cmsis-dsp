#!/bin/sh
# Build every variant for Cortex-M4F and Cortex-M55 and run it under QEMU with instruction counting.
# Run inside the Docker image: docker run --rm -v "$PWD/..:/w" -w /w/mcu sefr-mcu sh run.sh
set -e
for c in m4 m55; do for v in sefr lda gnb; do make -s CPU=$c VARIANT=$v; done; done
for f in build/*.elf; do echo "## $f"; arm-none-eabi-nm --size-sort -S -t d $f | awk '$3 ~ /[tT]/ {print $2+0, $4}'; done > ../results/sizes.txt
for c in m4 m55; do
  M=mps2-an386; [ $c = m55 ] && M=mps3-an547
  for v in sefr lda gnb; do
    echo "## $c $v"
    qemu-system-arm -M $M -nographic -monitor none -serial none -icount shift=0 \
      -semihosting-config enable=on,target=native -kernel build/${c}_$v.elf
  done
done > ../results/qemu_runs.txt 2>&1
python3 parse.py
