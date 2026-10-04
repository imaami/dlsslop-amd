// SPDX-License-Identifier: MIT
// Independent HIP test: no model weights, game or worker required.
#include "../backend/color_preserve.hpp"
#include "../backend/native_kernels.hpp"
#include <getopt.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
// The test's stream, events and buffers, released however it ends.
struct Resources {
    const dlsslop::hip::Api& api;
    dlsslop::hip::Handle stream = nullptr, start = nullptr, end = nullptr;
    void *original = nullptr, *raw = nullptr, *output = nullptr;
    ~Resources()
    {
        if (stream) api.hipStreamSynchronize(stream);
        for (auto event : {end, start})
            if (event) api.hipEventDestroy(event);
        for (void* buffer : {output, raw, original})
            if (buffer) api.hipFree(buffer);
        if (stream) api.hipStreamDestroy(stream);
    }
};

dlsslop::Result<void> run(const dlsslop::hip::Api& api, int device, const std::string& module)
{
    DLSSLOP_TRY(api.check(api.hipSetDevice(device), "select device"));
    Resources r{api};
    DLSSLOP_TRY(api.check(api.hipStreamCreate(&r.stream), "create stream"));
    const struct geometry g{1920,1080,1920,1152,1080,0,0,1920,1080};
    const std::size_t pixels=std::size_t(g.width)*g.height;
    std::vector<float> input(pixels*4),model(pixels*3),expected,actual(pixels*3);
    for(std::size_t p=0;p<pixels;++p) {
        for(unsigned c=0;c<3;++c) {
            input[p*4+c]=float((p*11+c*71)%1000)/999;
            model[p*3+c]=float((p*37+c*113)%1000)/999;
        }
        input[p*4+3]=1;
    }
    DLSSLOP_TRY(api.check(api.hipMalloc(&r.original,input.size()*sizeof(float)),"allocate test input"));
    DLSSLOP_TRY(api.check(api.hipMalloc(&r.raw,model.size()*sizeof(float)),"allocate test model"));
    DLSSLOP_TRY(api.check(api.hipMalloc(&r.output,model.size()*sizeof(float)),"allocate test output"));
    DLSSLOP_TRY(api.check(api.hipMemcpy(r.original,input.data(),input.size()*sizeof(float),1),"upload test reference"));
    DLSSLOP_TRY(api.check(api.hipMemcpy(r.raw,model.data(),model.size()*sizeof(float),1),"upload test model"));
    dlsslop::NativeKernels kernels(api,r.stream);
    DLSSLOP_TRY(kernels.load(module));
    // The kernel reads a neighbourhood of the model output, so it cannot run in place.
    if(const auto in_place=dlsslop::gpu_preserve_color(kernels,g,r.original,r.raw,r.raw,1);
       in_place || in_place.error().what.find("distinct input and output")==std::string::npos)
        return dlsslop::fail("GPU correction accepted an in-place output");
    for(unsigned reference=0;reference<2;++reference) {
        if(reference) {
            // Each apply reads the caller's reference as it is when the kernel runs.
            for(std::size_t p=0;p<pixels;++p)
                for(unsigned c=0;c<3;++c) input[p*4+c]=float((p*53+c*29)%1000)/999;
            DLSSLOP_TRY(api.check(api.hipMemcpy(r.original,input.data(),input.size()*sizeof(float),1),"replace test reference"));
        }
        for(float strength : {0.f,.25f,.5f,1.f}) {
            DLSSLOP_TRY(dlsslop::preserve_color(input.data(),model.data(),g,strength,expected));
            DLSSLOP_TRY(dlsslop::gpu_preserve_color(kernels,g,r.original,r.raw,r.output,strength));
            DLSSLOP_TRY(api.check(api.hipStreamSynchronize(r.stream),"finish correction"));
            DLSSLOP_TRY(api.check(api.hipMemcpy(actual.data(),r.output,actual.size()*sizeof(float),2),"read correction"));
            float worst=0;
            for(std::size_t i=0;i<actual.size();++i) {
                if(!std::isfinite(actual[i])) return dlsslop::fail("nonfinite GPU correction");
                worst=std::max(worst,std::fabs(actual[i]-expected[i]));
            }
            std::printf("reference=%u strength=%g max_abs_error=%.9g\n",reference,double(strength),double(worst));
            if(worst>2e-6f) return dlsslop::fail("GPU correction differs from CPU reference");
        }
    }
    DLSSLOP_TRY(api.check(api.hipEventCreate(&r.start),"create start event"));
    DLSSLOP_TRY(api.check(api.hipEventCreate(&r.end),"create end event"));
    DLSSLOP_TRY(api.check(api.hipEventRecord(r.start,r.stream),"record start"));
    for(unsigned i=0;i<20;++i) DLSSLOP_TRY(dlsslop::gpu_preserve_color(kernels,g,r.original,r.raw,r.output,1));
    DLSSLOP_TRY(api.check(api.hipEventRecord(r.end,r.stream),"record end"));
    DLSSLOP_TRY(api.check(api.hipEventSynchronize(r.end),"wait for timing"));
    float ms=0;
    DLSSLOP_TRY(api.check(api.hipEventElapsedTime(&ms,r.start,r.end),"measure correction"));
    std::printf("GPU correction mean_ms=%.6f over 20 runs; excludes inference\n",double(ms)/20);
    return {};
}
}

int main(int argc,char** argv)
{
    std::string module="assets/HIP/gfx1201/linux_native.hsaco";
    int device=-1;
    const option options[]={{"module",required_argument,nullptr,'m'}, {"device",required_argument,nullptr,'d'},
        {"help",no_argument,nullptr,'h'}, {nullptr,0,nullptr,0}};
    int code;
    while((code=getopt_long(argc,argv,"+m:d:h",options,nullptr))!=-1) {
        if(code=='h') {
            std::puts("Usage: color-gpu-test [OPTION]...\n"
                " -m, --module PATH  Kernel module (default: assets/HIP/gfx1201/linux_native.hsaco)\n"
                " -d, --device N     HIP device (default: auto, first gfx1201)\n"
                " -h, --help         Show help (default: off)\n"
                "Tests correction against CPU reference for two references, then times a 1080-tier kernel.\n"
                "No model assets required. Exit 77 means the module, HIP runtime or device is unavailable.");return 0;
        }
        if(code=='m') {module=optarg;continue;}
        if(code=='d') {
            char* end=nullptr;const long n=std::strtol(optarg,&end,10);
            if(!*optarg || *end || n<0 || n>1024) return 2;
            device=int(n);continue;
        }
        return 2;
    }
    if(optind!=argc || module.empty()) return 2;
    if(access(module.c_str(),R_OK)) {std::fprintf(stderr,"SKIP: cannot read module: %s\n",module.c_str());return 77;}
    const auto loaded=dlsslop::hip::load();
    if(!loaded) {std::fprintf(stderr,"SKIP: %s\n",loaded.error().what.c_str());return 77;}
    const auto& api=*loaded;
    // A runtime that loads but fails is a failure; only an absent device skips.
    int devices=0;
    const auto listed=[&]() -> dlsslop::Result<void> {
        DLSSLOP_TRY(api.check(api.hipInit(0),"initialize HIP"));
        DLSSLOP_TRY(api.check(api.hipGetDeviceCount(&devices),"enumerate devices"));
        for(int i=0;device<0 && i<devices;++i) {
            dlsslop::hip::DeviceProperties properties{};
            DLSSLOP_TRY(api.check(api.hipGetDevicePropertiesR0600(&properties,i),"device properties"));
            if(std::string(properties.gcnArchName).find("gfx1201")==0) device=i;
        }
        return {};
    }();
    if(!listed) {std::fprintf(stderr,"%s\n",listed.error().what.c_str());return 1;}
    if(!devices) {std::fprintf(stderr,"SKIP: no HIP devices\n");return 77;}
    if(device<0) {std::fprintf(stderr,"SKIP: no gfx1201 device\n");return 77;}
    const auto result=run(api,device,module);
    if(!result) {std::fprintf(stderr,"%s\n",result.error().what.c_str());return 1;}
    return 0;
}
