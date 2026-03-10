SRC	:= $(wildcard src/*.c) $(wildcard src/*.cpp)
OBJ	:= $(shell mkdir -p obj)
OBJ	:= $(patsubst src/%,obj/%,$(SRC:.c=.o))
OBJ	:= $(OBJ:.cpp=.o)
DEPS	:= $(OBJ:.o=.d)

EXPAND	?= 50

HIPCCC	:= hipcc -O3 -DNDEBUG --offload-arch=$(ARCH) -DEXPAND=$(EXPAND) -MMD -MP

ifndef ARCH
  $(error env var ARCH is not set)
endif

TARGET := vanity_torv3_rocm-$(ARCH)-e$(EXPAND)

all: $(TARGET)

$(TARGET): $(OBJ)
	hipcc -O3 -flto -DNDEBUG --offload-arch=$(ARCH) $^ -o $@

obj/%.o: src/%.c
	$(HIPCCC) -c $< -o $@

obj/%.o: src/%.cpp
	$(HIPCCC) -c $< -o $@

-include $(DEPS)

clean:
	rm -rf obj $(TARGET)

.PHONY: all clean
