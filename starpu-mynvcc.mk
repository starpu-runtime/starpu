
# Avoid using nvcc when making a coverity build, nvcc produces millions of
# lines of code which we don't want to analyze.  Instead, build dumb .o files
# containing empty functions.
V_mynvcc_ = $(V_mynvcc_$(AM_DEFAULT_VERBOSITY))
V_mynvcc_0 = @echo "  myNVCC  " $@;
V_mynvcc_1 = 
V_mynvcc = $(V_mynvcc_$(V))
.cu.o:
	@$(MKDIR_P) `dirname $@`
	$(V_mynvcc)grep 'extern *"C" *void *' $< | sed -ne 's/extern *"C" *void *\([a-zA-Z0-9_]*\) *(.*/void \1(void) {}/p' | $(CC) -x c - -o $@ -c
