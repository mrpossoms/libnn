$(eval OS := $(shell uname))

CC=gcc
CFLAGS=-g --std=c99 -D_XOPEN_SOURCE=500 -ftree-vectorize -O3 -march=skylake -ffast-math -Wno-implicit-function-declaration
SRCS=nn.c
INC=-I./src
LINK=-lm -lpthread
TST_SRC=nn_mat_mul nn_mat_func nn_conv_patch nn_conv_max_pool nn_conv nn_mat_load fc_model_test conv_model_test

TARGET=$(shell $(CC) -dumpmachine)

.PHONY all: static
static: $(addprefix obj/,$(SRCS:.c=.o))
	mkdir -p build/$(TARGET)/lib
	$(AR) rcs build/$(TARGET)/lib/libnn.a $^

bin/tests:
	mkdir -p bin/tests

obj:
	mkdir obj

build:
	mkdir build

obj/%.o: src/%.c obj build
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

install:
	cp src/nn.h /usr/local/include
	cp build/$(TARGET)/lib/*.a /usr/local/lib

tests: bin/tests
	@echo "Building tests..."
	@for source in $(TST_SRC); do\
		($(CC) $(INC) $(CFLAGS) src/tests/$$source.c  -o bin/tests/$${source%.*}.bin $(LINK)) || (exit 1);\
	done

test: tests
	@./test_runner.py

clean:
	@rm -rf obj bin build/$(TARGET) 
