showcheck:
	-cat $(TEST_LOGS) /dev/null
	! grep -q "ERROR: AddressSanitizer: " $(TEST_LOGS) /dev/null
	! grep -q "WARNING: AddressSanitizer: " $(TEST_LOGS) /dev/null
	! grep -q "ERROR: ThreadSanitizer: " $(TEST_LOGS) /dev/null
	! grep -q "WARNING: ThreadSanitizer: " $(TEST_LOGS) /dev/null
	RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i showcheck || RET=1 ; \
	done ; \
	exit $$RET
