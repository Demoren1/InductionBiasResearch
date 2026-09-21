MAIN := main
BUILD_DIR := build

.PHONY: all clean

all:
	mkdir -p $(BUILD_DIR)
	latexmk -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(BUILD_DIR) $(MAIN).tex
	cp $(BUILD_DIR)/$(MAIN).pdf $(MAIN).pdf

clean:
	latexmk -C -outdir=$(BUILD_DIR) $(MAIN).tex
