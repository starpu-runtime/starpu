recheck:
	RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i recheck || RET=1 ; \
	done ; \
	exit $$RET

showfailed:
	@RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i showfailed || RET=1 ; \
	done ; \
	exit $$RET

showcheck:
	RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i showcheck || RET=1 ; \
	done ; \
	exit $$RET

showsuite:
	RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i showsuite || RET=1 ; \
	done ; \
	exit $$RET
