# make CPU=m4|m55 VARIANT=base|sefr|lda|gnb
CPU ?= m4
VARIANT ?= base
DSP = ../third_party/CMSIS-DSP
ifeq ($(CPU),m4)
ARCH = -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard
else
ARCH = -mcpu=cortex-m55 -mfloat-abi=hard -DARM_MATH_HELIUM
endif
DEFS_base =
DEFS_sefr = -DWITH_SEFR
DEFS_lda = -DWITH_SEFR -DWITH_LDA
DEFS_gnb = -DWITH_GNB
CFLAGS = $(ARCH) -Os -ffunction-sections -fdata-sections -DARM_MATH_LOOPUNROLL -DDISABLEFLOAT16 \
  -I$(DSP)/Include -I$(DSP)/PrivateInclude -I../third_party/CMSIS_6/CMSIS/Core/Include -I../src $(DEFS_$(VARIANT))
DSPSRC = $(DSP)/Source/BasicMathFunctions/BasicMathFunctions.c $(DSP)/Source/StatisticsFunctions/StatisticsFunctions.c \
  $(DSP)/Source/SupportFunctions/SupportFunctions.c $(DSP)/Source/MatrixFunctions/MatrixFunctions.c \
  $(DSP)/Source/BayesFunctions/BayesFunctions.c $(DSP)/Source/FastMathFunctions/FastMathFunctions.c \
  $(DSP)/Source/CommonTables/CommonTables.c
OUT = build/$(CPU)_$(VARIANT).elf
$(OUT): bench.c startup.c ../src/arm_sefr_f32.c ../bench/gnb_train.c link.ld
	mkdir -p build
	arm-none-eabi-gcc $(CFLAGS) -nostartfiles --specs=nano.specs -Tlink.ld -Wl,--gc-sections -Wl,-Map=$(OUT:.elf=.map) \
	  bench.c startup.c ../src/arm_sefr_f32.c ../bench/gnb_train.c $(DSPSRC) -lm -o $@
