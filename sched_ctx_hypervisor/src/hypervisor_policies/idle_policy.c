#include "policy_utils.h"

struct hypervisor_policy idle_policy = {
	.resize = _resize_to_unknown_receiver
};
