// A HIP runtime stand-in for the worker's loader test: libhip-loader-stub.so
// needs libhip-loader-stub-dependency.so, which the test leaves unreachable.
#ifdef HIP_LOADER_STUB_DEPENDENCY
int hip_loader_stub_dependency() { return 0; }
#else
int hip_loader_stub_dependency();
int hip_loader_stub() { return hip_loader_stub_dependency(); }
#endif
