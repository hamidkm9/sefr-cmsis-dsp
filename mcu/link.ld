MEMORY { CODE (rx) : ORIGIN = 0x00000000, LENGTH = 512K  RAM (rwx) : ORIGIN = 0x20000000, LENGTH = 448K }
ENTRY(Reset_Handler)
SECTIONS {
  .text : { KEEP(*(.vectors)) *(.text*) *(.rodata*) . = ALIGN(4); } > CODE
  .ARM.exidx : { *(.ARM.exidx*) } > CODE
  .data : { *(.data*) . = ALIGN(4); } > RAM
  .bss (NOLOAD) : { __bss_start = .; *(.bss*) *(COMMON) . = ALIGN(8); __bss_end = .; } > RAM
  __stack_top = ORIGIN(RAM) + LENGTH(RAM);
  end = __bss_end;
}
