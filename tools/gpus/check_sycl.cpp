#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <string>

int main()
{
	try
	{
		// Enumerate all SYCL platforms
		auto platforms = sycl::platform::get_platforms();
		for (const auto& plat : platforms)
		{
			auto devices = plat.get_devices();
			for (size_t i = 0; i < devices.size(); i++)
			{
				const auto& dev = devices[i];
				std::string backend;
				switch (plat.get_backend())
				{
				case sycl::backend::opencl:
					backend = "opencl";
					break;
				case sycl::backend::ext_oneapi_level_zero:
					backend = "level_zero";
					break;
				default:
					backend = "unknown";
					break;
				}

				std::string type;
				if (dev.is_cpu())
					type = "cpu";
				else if (dev.is_gpu())
					type = "gpu";
				else if (dev.is_accelerator())
					type = "accelerator";
				else
					type = "unknown";

				std::cout << "[" << backend << ":" << type << "]"
					  << "[" << backend << ":" << i << "] ";
				std::cout << plat.get_info<sycl::info::platform::name>() << ", "
					  << dev.get_info<sycl::info::device::name>() << " "
					  << dev.get_info<sycl::info::device::driver_version>()
					  << "\n";
			}
		}
	}
	catch (const sycl::exception& e)
	{
		std::cerr << "SYCL exception: " << e.what() << "\n";
		return 1;
	}
	return 0;
}
