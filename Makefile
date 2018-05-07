$(eval OS := $(shell uname))

CC=gcc
CFLAGS=-g --std=c99 -D_XOPEN_SOURCE=500 -ftree-vectorize
SRCS=nn.c
INC=-I./src
LINK=-lm -lpthread
TST_SRC=nn_mat_mul nn_mat_func nn_conv_patch nn_conv_max_pool nn_conv nn_mat_load fc_model_test


.PHONY all: static
static: $(addprefix obj/,$(SRCS:.c=.o))
	$(AR) rcs lib/libnn.a $^

bin/tests:
	mkdir -p bin/tests

obj:
	mkdir obj

lib:
	mkdir lib

obj/%.o: src/%.c obj lib
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

install:
	cp src/nn.h /usr/local/include
	cp lib /usr/local/lib

tests: bin/tests
	@echo "Building tests..."
	@for source in $(TST_SRC); do\
		($(CC) $(INC) $(CFLAGS) src/tests/$$source.c  -o bin/tests/$${source%.*}.bin $(LINK)) || (exit 1);\
	done

test: tests
	@./test_runner.py

clean:
	@rm -rf obj bin lib
