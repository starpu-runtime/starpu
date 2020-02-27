showfailed:
	@! grep "^FAIL " $(TEST_LOGS) /dev/null
	@RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i showfailed || RET=1 ; \
	done ; \
	exit $$RET

showcheck:
	-cat $(TEST_LOGS) /dev/null
	! grep -q "ERROR: AddressSanitizer: " $(TEST_LOGS) /dev/null
	! grep -q "WARNING: AddressSanitizer: " $(TEST_LOGS) /dev/null
	! grep -q "ERROR: ThreadSanitizer: " $(TEST_LOGS) /dev/null
	! grep -q "WARNING: ThreadSanitizer: " $(TEST_LOGS) /dev/null
	! grep -q "ERROR: LeakSanitizer: " $(TEST_LOGS) /dev/null
	! grep -q "WARNING: LeakSanitizer: " $(TEST_LOGS) /dev/null
	! grep -q " runtime error: " $(TEST_LOGS) /dev/null
	RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i showcheck || RET=1 ; \
	done ; \
	exit $$RET

showsuite:
	-cat $(TEST_SUITE_LOG) /dev/null
	@! grep -q "ERROR: AddressSanitizer: " $(TEST_SUITE_LOG) /dev/null
	@! grep -q "WARNING: AddressSanitizer: " $(TEST_SUITE_LOG) /dev/null
	@! grep -q "ERROR: ThreadSanitizer: " $(TEST_SUITE_LOG) /dev/null
	@! grep -q "WARNING: ThreadSanitizer: " $(TEST_SUITE_LOG) /dev/null
	@! grep -q "ERROR: LeakSanitizer: " $(TEST_SUITE_LOG) /dev/null
	@! grep -q "WARNING: LeakSanitizer: " $(TEST_SUITE_LOG) /dev/null
	@! grep -q " runtime error: " $(TEST_SUITE_LOG) /dev/null
	RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i showsuite || RET=1 ; \
	done ; \
	exit $$RET
