#include "spu_coherency.h"

/* the lock is a copy in local store */
/*
 *  NB : the ea_lock may be on a LS !
 */
void release_lock(data_lock *ls_lock)
{
	uintptr_t ea_lock = ls_lock->ea_taken;

	/* overwrite the value so that it should be FREE */
	uint32_t free __attribute__ ((aligned(16)));
	free= FREE; 

	mfc_put (&free, ea_lock, sizeof(uint32_t), COHERENCY_TAG, 0, 0);
	mfc_write_tag_mask(1 << COHERENCY_TAG);
	spu_mfcstat(MFC_TAG_UPDATE_ALL);
}

/*
 * This implementation is inspired by the Cell Handbook v1.1 page 595
 */
void take_lock(data_lock *ls_lock)
{
	uintptr_t ea_taken = ls_lock->ea_taken;

	/* we take a buffer aligned on a PPE cache line */
	volatile uint8_t buf[128] __attribute__ ((aligned(128)));
	volatile uint32_t *lock_ls_ptr;
	uint32_t offset;


	/* we only lock data per 128B as this is size of PPE cache lines */
	uintptr_t ea_lock_aligned = (ea_taken + 127) & ~127;

	offset = ea_taken & 127;
	lock_ls_ptr  = (volatile uint32_t *) (buf + offset);
	
	/* first we only pay attention to the lock lost events */
	uint32_t events;
	uint32_t event_mask;
	event_mask = spu_readch(SPU_RdEventMask);

	/* if there were pending events */
	spu_writech(SPU_WrEventMask, 0);
	if (spu_readchcnt(SPU_RdEventStat)) {
		spu_writech(SPU_WrEventAck, spu_readch(SPU_RdEventStat));
	}
	
	/* we monitor only lost events */
	spu_writech(SPU_WrEventMask, MFC_LLR_LOST_EVENT);

	uint32_t status;
	do {
		/* fetch the cache line AND reserve it */
		mfc_getllar(buf, ea_lock_aligned, ATOMIC_COHERENCY_TAG, 0);
		(void)spu_readch(MFC_RdAtomicStat);

		if (*lock_ls_ptr == FREE) {
			/* the lock is already taken, wait for an event that 
 			   acknowledge some activity on the cache line */
			events = spu_readch(SPU_RdEventStat);
			spu_writech(SPU_WrEventAck, events);

			status = MFC_PUTLLC_STATUS;
		} else {
			/* no one holds the lock yet */

			/* we modify the LS buffer and try to commit it */
			*lock_ls_ptr = TAKEN;
			mfc_putllc(buf, ea_lock_aligned, 
				ATOMIC_COHERENCY_TAG, 0);

			/* did the SPU lost the reservation in between ? */
			status = spu_readch(MFC_RdAtomicStat) 
					& MFC_PUTLLC_STATUS;	
		}
		
	} while (status);

	/* restore the event mask that we saved */
	spu_writech(SPU_WrEventMask, event_mask);
}

/* we do have the same code for PPU and SPU in the case of a Cell */
#include "../coherency_common.c"
