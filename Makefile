CC      ?= cc
CFLAGS  ?= -O2 -march=x86-64-v3 -Wall -Wextra
LDLIBS   = -lnuma

stress: main.c stress_kernel.S stress.h
	$(CC) $(CFLAGS) -pthread main.c stress_kernel.S $(LDLIBS) -o $@

clean:
	rm -f stress

.PHONY: clean
