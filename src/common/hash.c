#include <common/hash.h>

#define CRC32C_POLY_BE 0x1EDC6F41

static inline uint32_t crc32_be_8(uint8_t inputbyte, uint32_t inputcrc)
{
	unsigned i;
	uint32_t crc;

	crc = inputcrc ^ (inputbyte << 24);
	for (i = 0; i < 8; i++)
		crc = (crc << 1) ^ ((crc & 0x80000000) ? CRC32C_POLY_BE : 0);

	return crc;
}

uint32_t crc32_be(uint32_t input, uint32_t inputcrc)
{
	uint8_t *p = (uint8_t *)&input;

	uint32_t crc = inputcrc;

	crc = crc32_be_8(p[0], crc);
	crc = crc32_be_8(p[1], crc);
	crc = crc32_be_8(p[2], crc);
	crc = crc32_be_8(p[3], crc);

	return crc;
}
