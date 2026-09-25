// SPDX-License-Identifier: MIT
#pragma once
#include "../common/control_settings.h"
#include <algorithm>
#include <cmath>

namespace dlsslop_gui {

inline bool logarithmicSlider(const dlsslop_control::Setting& setting)
{
    return setting.isFloat && setting.minimum > 0 && setting.maximum / setting.minimum >= 1000;
}

inline int sliderPosition(const dlsslop_control::Setting& setting, double value)
{
    if (!setting.isFloat) return static_cast<int>(value);
    const double fraction = logarithmicSlider(setting) ?
        std::log(value / setting.minimum) / std::log(setting.maximum / setting.minimum) :
        (value - setting.minimum) / (setting.maximum - setting.minimum);
    return static_cast<int>(std::lround(std::clamp(fraction, 0.0, 1.0) * 10000.0));
}

inline double sliderValue(const dlsslop_control::Setting& setting, int position)
{
    if (!setting.isFloat) return position;
    const double fraction = position / 10000.0;
    return logarithmicSlider(setting) ? setting.minimum * std::pow(setting.maximum / setting.minimum, fraction) :
        setting.minimum + (setting.maximum - setting.minimum) * fraction;
}

} // namespace dlsslop_gui
