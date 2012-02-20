#include "policy_utils.h"

struct hypervisor_policy app_driven_policy = {
	.resize = _resize_to_unknown_receiver
};
