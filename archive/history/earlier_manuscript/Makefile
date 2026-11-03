MAIN := main
BUILD_DIR := build

# Use the optional project-local TeX distribution when it is installed.
LOCAL_TEX_BIN := $(wildcard $(abspath ../.tools/TinyTeX/bin)/*)
ifneq ($(LOCAL_TEX_BIN),)
export PATH := $(LOCAL_TEX_BIN):$(PATH)
endif
LATEXMK ?= $(if $(LOCAL_TEX_BIN),$(firstword $(LOCAL_TEX_BIN))/latexmk,latexmk)

.PHONY: all clean

all:
	mkdir -p $(BUILD_DIR)
	$(LATEXMK) -pdf -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error \
		-outdir=$(BUILD_DIR) $(MAIN).tex
	cp $(BUILD_DIR)/$(MAIN).pdf $(MAIN).pdf

clean:
	$(LATEXMK) -C -outdir=$(BUILD_DIR) $(MAIN).tex
