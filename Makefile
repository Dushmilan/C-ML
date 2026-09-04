CC=gcc
CFLAGS=-Wall -Wextra -I.
SRC=tensor.c ops.c Autograd.c memory_pool.c optimizer.c Module/module.c
test: tests/test_all.c $(SRC)
	$(CC) $(CFLAGS) $^ -lm -o /tmp/test_all && /tmp/test_all