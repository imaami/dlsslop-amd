#include "../backend/color_preserve.h"
#include <cstdio>
#include <limits>

static void check(bool value, const char* message)
{ if (!value) throw std::runtime_error(message); }
static float luma(const float* rgb) { return .2126f*rgb[0]+.7152f*rgb[1]+.0722f*rgb[2]; }
int main()
{
    try {
        dlsslop::Geometry g{8,8,8,8,8,8,1,1,6,6};
        std::vector<float> original(8*8*4, .4f), raw(8*8*3), result, half;
        for (unsigned p=0; p<64; ++p) {
            raw[p*3]=.6f; raw[p*3+1]=.4f; raw[p*3+2]=.2f;
        }
        dlsslop::preserve_color(original.data(),raw.data(),g,0,result);
        check(!std::memcmp(raw.data(),result.data(),raw.size()*sizeof(float)),"disabled not bit identical");
        dlsslop::preserve_color(original.data(),raw.data(),g,1,result);
        dlsslop::preserve_color(original.data(),raw.data(),g,.5f,half);
        for (unsigned y=0; y<8; ++y) for(unsigned x=0; x<8; ++x) {
            const unsigned p=(y*8+x)*3;
            if(x<1 || x>6 || y<1 || y>6) {
                check(!std::memcmp(raw.data()+p,result.data()+p,3*sizeof(float)),"padding modified");
                continue;
            }
            check(std::fabs(result[p]-result[p+2])<1e-6f,"uniform cast not removed");
            check(std::fabs(luma(raw.data()+p)-luma(result.data()+p))<1e-6f,"luma changed");
            check(std::fabs((half[p]-half[p+2])-.2f)<1e-6f,"strength not proportional");
        }
        // Repeated warm bias must be removed against the same original reference.
        for(unsigned pass=0; pass<3; ++pass) {
            for(unsigned p=0;p<64;++p) { raw[p*3]=result[p*3]+.04f; raw[p*3+1]=result[p*3+1]; raw[p*3+2]=result[p*3+2]-.04f; }
            dlsslop::preserve_color(original.data(),raw.data(),g,1,result);
            check(std::fabs(result[3*27]-result[3*27+2])<1e-6f,"repeated cast accumulated");
        }
        // Achromatic high-frequency detail is retained, even on colored reference.
        for(unsigned p=0;p<64;++p) {
            original[p*4]=.3f; original[p*4+1]=.4f; original[p*4+2]=.5f;
            const float light=p%2 ? .1f : -.1f;
            for(unsigned c=0;c<3;++c) raw[p*3+c]=original[p*4+c]+light;
        }
        dlsslop::preserve_color(original.data(),raw.data(),g,1,result);
        for(unsigned i=0;i<raw.size();++i) check(std::fabs(result[i]-raw[i])<1e-6f,"achromatic detail lost");
        // Out-of-gamut corrected chroma compresses without changing in-range luma.
        for(unsigned p=0;p<64;++p) { original[p*4]=1;original[p*4+1]=0;original[p*4+2]=0;raw[p*3]=raw[p*3+1]=raw[p*3+2]=.8f; }
        dlsslop::preserve_color(original.data(),raw.data(),g,1,result);
        for(unsigned c=0;c<3;++c) check(result[27*3+c]>=0 && result[27*3+c]<=1,"gamut excursion");
        check(std::fabs(luma(result.data()+27*3)-.8f)<1e-6f,"gamut mapping changed luma");
        bool rejected=false;
        try { dlsslop::preserve_color(original.data(),raw.data(),g,std::numeric_limits<float>::quiet_NaN(),result); }
        catch(const std::invalid_argument&) { rejected=true; }
        check(rejected,"NaN strength accepted");
        std::puts("Color preservation: bypass, strength, luma, detail, padding, repeated bias, gamut and validation passed");
    } catch(const std::exception& error) { std::fprintf(stderr,"%s\n",error.what());return 1; }
}
