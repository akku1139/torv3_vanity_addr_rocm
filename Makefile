SRC := $(wildcard src/*.c) $(wildcard src/*.cpp)

ifndef ARCH
  $(error env ver ARCH is not set)
endif

TARGET := vanity_torv3_rocm-$(ARCH)

$(TARGET): $(SRC)
	hipcc -O3 -flto -DNDEBUG --offload-arch=$(ARCH) $^ -o $@
