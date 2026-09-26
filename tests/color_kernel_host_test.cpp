// Exercises the actual kernel entry point on the host. Not a HIP execution test.
#include "../backend/color_preserve.h"
#include <cstdio>
static unsigned group_id, item_id;
static unsigned __builtin_amdgcn_workgroup_id_x() { return group_id; }
static unsigned __builtin_amdgcn_workitem_id_x() { return item_id; }
#include "../backend/color_gpu.hip"
int main()
{
    const dlsslop::Geometry g{37,19,37,19,37,19,2,1,32,17};
    const unsigned pixels=g.width*g.height;
    std::vector<float> original(pixels*4), raw(pixels*3), actual(pixels*3+8,12345), expected;
    for(unsigned p=0;p<pixels;++p) {
        for(unsigned c=0;c<3;++c) {
            original[p*4+c]=float((p*11+c*71)%1000)/999;
            raw[p*3+c]=float((p*37+c*113)%1000)/999;
        }
        original[p*4+3]=1;
    }
    for(float strength : {0.f,.25f,.5f,1.f}) {
        dlsslop::preserve_color(original.data(),raw.data(),g,strength,expected);
        for(group_id=0;group_id<(pixels+255)/256;++group_id)
            for(item_id=0;item_id<256;++item_id)
                dlsslop_preserve_color(original.data(),raw.data(),actual.data(),g,strength);
        if(std::memcmp(expected.data(),actual.data(),expected.size()*sizeof(float))) {
            std::fprintf(stderr,"kernel entry disagrees with reference\n");return 1;
        }
        for(unsigned i=pixels*3;i<actual.size();++i)
            if(actual[i]!=12345) {std::fprintf(stderr,"kernel tail overrun\n");return 1;}
    }
    std::puts("Host kernel entry: four strengths, fitted boundaries and partial workgroup passed");
}
