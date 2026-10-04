#include "../backend/color_preserve.hpp"
#include "golden.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

static void check(bool value, const char* message)
{
    if (value) return;
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}
static void preserve(const std::vector<float>& original, const std::vector<float>& raw, const dlsslop::Geometry& g,
                     float strength, std::vector<float>& result)
{
    check(bool(dlsslop::preserve_color(original.data(), raw.data(), g, strength, result)),
          "valid color preservation refused");
}
static float luma(const float* rgb) { return .2126f*rgb[0]+.7152f*rgb[1]+.0722f*rgb[2]; }
// Corrects a uniform frame of one original and one model colour; returns the centre pixel.
static std::array<float,3> uniform(const float* original_rgb, const float* model_rgb, float strength)
{
    const dlsslop::Geometry g{8,8,8,8,8,1,1,6,6};
    std::vector<float> original(8*8*4, 1), raw(8*8*3), result;
    for (unsigned p=0; p<64; ++p) for (unsigned c=0; c<3; ++c) {
        original[p*4+c]=original_rgb[c];
        raw[p*3+c]=model_rgb[c];
    }
    preserve(original,raw,g,strength,result);
    return {result[27*3], result[27*3+1], result[27*3+2]};
}
// The reference's output at four strengths for a random letterboxed fixture, a
// little outside 0..1, and a model that drifted from it, bit for bit
// (golden.hpp). 3440x1440 at the 720 tier: letterboxed, and padded below.
static void goldens()
{
    const dlsslop::Geometry g{3440,1440,1280,768,720,0,92,1280,536};
    const std::size_t pixels=std::size_t(g.width)*g.height;
    std::vector<float> original(pixels*4), model(pixels*3), result;
    for (std::size_t p=0; p<pixels; ++p) {
        for (unsigned c=0; c<3; ++c) {
            original[p*4+c]=golden::unit(1,p*3+c)*1.25f-.125f;
            model[p*3+c]=original[p*4+c]+(golden::unit(2,p*3+c)-.5f)*.25f;
        }
        original[p*4+3]=1;
    }
    const struct { float strength; std::uint64_t golden; } strengths[]={
        {0,0xf44c4ac95f2260f1u}, {.25f,0x928f9929da24f922u},
        {.5f,0xfadb9dc5a658363bu}, {1,0x57b4e5d13a255e46u}};
    unsigned moved=0;
    char name[32];
    for (const auto& s : strengths) {
        preserve(original,model,g,s.strength,result);
        std::snprintf(name,sizeof name,"preserve_color %g",double(s.strength));
        moved+=!golden::check(name,result.data(),result.size()*sizeof(float),s.golden);
    }
    check(!moved,"a golden moved");
}
int main()
{
    dlsslop::Geometry g{8,8,8,8,8,1,1,6,6};
    std::vector<float> original(8*8*4, .4f), raw(8*8*3), result, half;
    for (unsigned p=0; p<64; ++p) {
        raw[p*3]=.6f; raw[p*3+1]=.4f; raw[p*3+2]=.2f;
    }
    preserve(original,raw,g,0,result);
    check(!std::memcmp(raw.data(),result.data(),raw.size()*sizeof(float)),"disabled not bit identical");
    preserve(original,raw,g,1,result);
    preserve(original,raw,g,.5f,half);
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
        preserve(original,raw,g,1,result);
        check(std::fabs(result[3*27]-result[3*27+2])<1e-6f,"repeated cast accumulated");
    }
    // Achromatic high-frequency detail is retained, even on colored reference.
    for(unsigned p=0;p<64;++p) {
        original[p*4]=.3f; original[p*4+1]=.4f; original[p*4+2]=.5f;
        const float light=p%2 ? .1f : -.1f;
        for(unsigned c=0;c<3;++c) raw[p*3+c]=original[p*4+c]+light;
    }
    preserve(original,raw,g,1,result);
    for(unsigned i=0;i<raw.size();++i) check(std::fabs(result[i]-raw[i])<1e-6f,"achromatic detail lost");
    // Out-of-gamut corrected chroma compresses without changing in-range luma.
    for(unsigned p=0;p<64;++p) { original[p*4]=1;original[p*4+1]=0;original[p*4+2]=0;raw[p*3]=raw[p*3+1]=raw[p*3+2]=.8f; }
    preserve(original,raw,g,1,result);
    for(unsigned c=0;c<3;++c) check(result[27*3+c]>=0 && result[27*3+c]<=1,"gamut excursion");
    check(std::fabs(luma(result.data()+27*3)-.8f)<1e-6f,"gamut mapping changed luma");
    // The model's own excursions outside [0,1] are not the correction's to undo: with no
    // drift the result is the model, bit for bit, at any strength.
    const float excursions[][3]={{1.2f,.9f,.9f},{-.05f,.02f,.6f}};
    for (const auto& model : excursions) for (float strength : {1e-6f,.5f,1.f}) {
        const auto kept=uniform(model,model,strength);
        check(!std::memcmp(kept.data(),model,sizeof(model)),"model excursion changed without drift");
    }
    // Tuning keeps headroom above 1; a negligible strength must not clip it.
    const float tuned_original[]={.9f,.5f,.5f}, tuned[]={1.1f,.5f,.5f};
    const auto slight=uniform(tuned_original,tuned,1e-6f);
    for (unsigned c=0; c<3; ++c) check(std::fabs(slight[c]-tuned[c])<1e-6f,"negligible strength jumped");
    // Saturated SDR blue whose model undershoots red: chroma only moves toward the original's,
    // in proportion to strength, and luma stays.
    const float blue[]={0,.05f,.6f}, undershoot[]={-.02f,.05f,.6f};
    for (float strength : {.01f,.5f,1.f}) {
        const auto moved=uniform(blue,undershoot,strength);
        for (unsigned c=0; c<3; ++c) {
            const float from=undershoot[c]-luma(undershoot), to=blue[c]-luma(blue);
            const float chroma=moved[c]-luma(moved.data());
            check(std::fabs(chroma-(from+strength*(to-from)))<1e-6f,"chroma moved away from original");
        }
        check(std::fabs(luma(moved.data())-luma(undershoot))<1e-6f,"SDR correction changed luma");
    }
    // An HDR highlight keeps its headroom: out-of-range luma takes the full correction.
    const float highlight_original[]={2.2f,1.9f,1.9f}, highlight[]={2,2,2};
    const auto lit=uniform(highlight_original,highlight,1);
    const float drift[]={-.2f,.1f,.1f};
    for (unsigned c=0; c<3; ++c)
        check(std::fabs(lit[c]-(highlight[c]-drift[c]+luma(drift)))<1e-5f,"HDR highlight lost its correction");
    check(std::fabs(luma(lit.data())-2)<1e-6f,"HDR correction changed luma");
    const auto nan=dlsslop::preserve_color(original.data(),raw.data(),g,std::numeric_limits<float>::quiet_NaN(),result);
    check(!nan && !nan.error().rejected && nan.error().what=="color preservation must be finite and within 0..1",
          "NaN strength accepted");
    goldens();
    std::puts("Color preservation: bypass, strength, luma, detail, padding, repeated bias, gamut, "
              "model excursions, headroom, validation and goldens passed");
}
