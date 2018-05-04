#include "test.h"

int succeed(void)
{
	return 0;
}

TEST_BEGIN
	.name = "Blank Test",
	.description = "This test does nothing but pass.",
	.run = succeed,
TEST_END
