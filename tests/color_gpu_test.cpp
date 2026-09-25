// SPDX-License-Identifier: MIT
// Independent HIP test: no model weights, game or worker required.
#include "../backend/color_gpu.h"
#include "../backend/color_preserve.h"
#include <getopt.h>
#include <cstdio>
#include <memory>

int main(int argc,char** argv)
{
    std::string module="assets/HIP/gfx1201/linux_color.hsaco";
    int device=-1;
    const option options[]={{"module",required_argument,nullptr,'m'}, {"device",required_argument,nullptr,'d'},
        {"help",no_argument,nullptr,'h'}, {nullptr,0,nullptr,0}};
    int code;
    while((code=getopt_long(argc,argv,"+m:d:h",options,nullptr))!=-1) {
        if(code=='h') {
            std::puts("Usage: color-gpu-test [OPTION]...\n"
                " -m, --module PATH  Kernel module (default: assets/HIP/gfx1201/linux_color.hsaco)\n"
                " -d, --device N     HIP device (default: auto, first gfx1201)\n"
                " -h, --help         Show help (default: off)\n"
                "Tests correction against CPU reference, then times a 1080-tier kernel.\n"
                "No model assets required. Exit 77 means HIP runtime/device unavailable.");return 0;
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
    std::unique_ptr<hip_probe::Api> holder;
    try { holder=std::make_unique<hip_probe::Api>(); }
    catch(const std::exception& e) {std::fprintf(stderr,"SKIP: %s\n",e.what());return 77;}
    auto& api=*holder;
    hip_probe::Handle stream=nullptr,start=nullptr,end=nullptr;
    void* original=nullptr;void* raw=nullptr;
    int result=0;
    try {
        api.Check(api.hipInit(0),"initialize HIP");
        int devices=0;api.Check(api.hipGetDeviceCount(&devices),"enumerate devices");
        if(!devices) {std::fprintf(stderr,"SKIP: no HIP devices\n");return 77;}
        if(device<0) {
            for(int i=0;i<devices;++i) {
                const auto properties=api.Properties(i);
                if(std::string(properties.gcnArchName).find("gfx1201")==0) {device=i;break;}
            }
            if(device<0) {std::fprintf(stderr,"SKIP: no gfx1201 device\n");return 77;}
        }
        api.Check(api.hipSetDevice(device),"select device");
        api.Check(api.hipStreamCreate(&stream),"create stream");
        const dlsslop::Geometry g{1920,1080,1920,1152,1920,1080,0,0,1920,1080};
        const std::size_t pixels=std::size_t(g.width)*g.height;
        std::vector<float> input(pixels*4),model(pixels*3),expected,actual(pixels*3);
        for(std::size_t p=0;p<pixels;++p) {
            for(unsigned c=0;c<3;++c) {
                input[p*4+c]=float((p*11+c*71)%1000)/999;
                model[p*3+c]=float((p*37+c*113)%1000)/999;
            }
            input[p*4+3]=1;
        }
        api.Check(api.hipMalloc(&original,input.size()*sizeof(float)),"allocate test input");
        api.Check(api.hipMalloc(&raw,model.size()*sizeof(float)),"allocate test output");
        api.Check(api.hipMemcpy(original,input.data(),input.size()*sizeof(float),1),"upload test reference");
        api.Check(api.hipMemcpy(raw,model.data(),model.size()*sizeof(float),1),"upload test model");
        {
            dlsslop::GpuColor color(api,stream,module,g.width,g.height);
            color.begin(original,g);
            // Subsequent input reuse must not overwrite the captured reference.
            api.Check(api.hipMemsetAsync(original,0,input.size()*sizeof(float),stream),"reuse source buffer after reference capture");
            for(float strength : {0.f,.25f,.5f,1.f}) {
                dlsslop::preserve_color(input.data(),model.data(),g,strength,expected);
                void* output=color.apply(raw,g,strength);
                api.Check(api.hipStreamSynchronize(stream),"finish correction");
                api.Check(api.hipMemcpy(actual.data(),output,actual.size()*sizeof(float),2),"read correction");
                float worst=0;
                for(std::size_t i=0;i<actual.size();++i) {
                    if(!std::isfinite(actual[i])) throw std::runtime_error("nonfinite GPU correction");
                    worst=std::max(worst,std::fabs(actual[i]-expected[i]));
                }
                std::printf("strength=%g max_abs_error=%.9g\n",double(strength),double(worst));
                if(worst>2e-6f) throw std::runtime_error("GPU correction differs from CPU reference");
            }
            api.Check(api.hipEventCreate(&start),"create start event");
            api.Check(api.hipEventCreate(&end),"create end event");
            api.Check(api.hipEventRecord(start,stream),"record start");
            for(unsigned i=0;i<20;++i) color.apply(raw,g,1);
            api.Check(api.hipEventRecord(end,stream),"record end");
            api.Check(api.hipEventSynchronize(end),"wait for timing");
            float ms=0;api.Check(api.hipEventElapsedTime(&ms,start,end),"measure correction");
            std::printf("GPU correction mean_ms=%.6f over 20 runs; excludes frame-reference copy and inference\n",double(ms)/20);
        }
    } catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());result=1;}
    if(stream) api.hipStreamSynchronize(stream);
    if(end) api.hipEventDestroy(end);
    if(start) api.hipEventDestroy(start);
    if(raw) api.hipFree(raw);
    if(original) api.hipFree(original);
    if(stream) api.hipStreamDestroy(stream);
    return result;
}
