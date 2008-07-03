#include "driver_spu_spu.h"

unsigned spuindex;

int main(uint64_t id __attribute__ ((unused)),
	uint64_t argspu __attribute__ ((unused)),
	uint64_t ppu_envp __attribute__ ((unused)))
{
	/* do nothing for now ! */

	/* retrieve the arguments */
	unsigned tag = 0;

	spu_init_arguments ls_arguments;
	mfc_get(&ls_arguments, argspu, sizeof(ls_arguments), tag, 0, 0);

	/* wait for the header */
	mfc_write_tag_mask(1 << tag);
	spu_mfcstat(MFC_TAG_UPDATE_ALL);

	spuindex = ls_arguments.deviceid;

	/* notify the PPU that it can go on by setting the proper flag */
	uintptr_t ea_ready_flag;
	ea_ready_flag = (uintptr_t)ls_arguments.ea_ready_flag;

	uint32_t unity = 1;
	mfc_put (&unity, ea_ready_flag, sizeof(uint32_t), tag, 0, 0);
	mfc_write_tag_mask(1 << tag);
	spu_mfcstat(MFC_TAG_UPDATE_ALL);

	return 0;
}
