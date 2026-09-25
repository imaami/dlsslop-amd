// SPDX-License-Identifier: MIT
#include "absolute_slider.h"
#include "slider_scale.h"
#include "theme.h"
#include <QApplication>
#include <QFrame>
#include <QSignalSpy>
#include <QStyleFactory>
#include <QTest>
#include <cstdio>
#include <stdexcept>
#include <string_view>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

class Slider final : public dlsslop_gui::AbsoluteSlider {
public:
    using AbsoluteSlider::AbsoluteSlider;

    // Ask the installed style where a value is painted, independently of the
    // widget's pointer-to-value calculation.
    QPoint pointAt(int position) const
    {
        QStyleOptionSlider option;
        initStyleOption(&option);
        option.sliderPosition = position;
        return style()->subControlRect(QStyle::CC_Slider, &option,
                                       QStyle::SC_SliderHandle, this).center();
    }
};

void moveHeld(Slider& slider, const QPoint& point)
{
    QMouseEvent move(QEvent::MouseMove, point, slider.mapToGlobal(point),
                     Qt::NoButton, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(&slider, &move);
}

const dlsslop_control::Setting& setting(std::string_view name)
{
    for (const auto& value : dlsslop_control::kSettings)
        if (value.name == name) return value;
    throw std::runtime_error("missing test setting");
}

void interaction(Slider& slider)
{
    slider.setRange(0, 100);
    slider.setValue(10);
    QSignalSpy pressed(&slider, &QSlider::sliderPressed);
    QSignalSpy released(&slider, &QSlider::sliderReleased);
    QSignalSpy changed(&slider, &QSlider::valueChanged);
    QTest::mousePress(&slider, Qt::LeftButton, Qt::NoModifier, slider.pointAt(75));
    require(slider.value() == 75, "track press must immediately jump to its position");
    require(slider.isSliderDown() && pressed.count() == 1, "press must start a drag");
    require(changed.count() == 1, "press must publish one immediate value change");
    moveHeld(slider, slider.pointAt(25));
    require(slider.value() == 25, "drag after a track click must follow the pointer");
    QTest::mouseRelease(&slider, Qt::LeftButton, Qt::NoModifier, slider.pointAt(60));
    require(slider.value() == 60 && !slider.isSliderDown() && released.count() == 1,
            "release must publish its final position and end the drag once");
    QTest::mousePress(&slider, Qt::LeftButton, Qt::NoModifier, slider.pointAt(60));
    moveHeld(slider, QPoint(slider.width() + 40, slider.height() / 2));
    require(slider.value() == 100, "drag beyond the right edge must reach maximum");
    moveHeld(slider, QPoint(-40, slider.height() / 2));
    require(slider.value() == 0, "drag beyond the left edge must reach minimum");
    QTest::mouseRelease(&slider, Qt::LeftButton, Qt::NoModifier, slider.pointAt(0));

    slider.setTracking(false);
    QTest::mousePress(&slider, Qt::LeftButton, Qt::NoModifier, slider.pointAt(50));
    require(slider.value() == 0 && slider.sliderPosition() == 50, "disabled tracking must defer value changes");
    QTest::mouseRelease(&slider, Qt::LeftButton, Qt::NoModifier, slider.pointAt(75));
    require(slider.value() == 75, "disabled tracking must commit on release");
    slider.setTracking(true);

    require(slider.hasFocus(), "mouse interaction must retain keyboard focus");
    QTest::keyClick(&slider, Qt::Key_Home);
    QTest::keyClick(&slider, Qt::Key_Right);
    require(slider.value() == 1, "arrow key must retain its native single step");
    QTest::keyClick(&slider, Qt::Key_PageUp);
    require(slider.value() == 11, "Page Up must retain its native page step");
    QTest::keyClick(&slider, Qt::Key_End);
    require(slider.value() == 100, "End must reach maximum");
}

void directions(Slider& slider)
{
    for (const auto orientation : {Qt::Horizontal, Qt::Vertical}) {
        slider.setOrientation(orientation);
        slider.resize(orientation == Qt::Horizontal ? QSize(420, 40) : QSize(40, 420));
        for (const auto direction : {Qt::LeftToRight, Qt::RightToLeft}) {
            slider.setLayoutDirection(direction);
            for (const bool inverted : {false, true}) {
                slider.setInvertedAppearance(inverted);
                for (const int value : {0, 25, 75, 100}) {
                    QTest::mouseClick(&slider, Qt::LeftButton, Qt::NoModifier, slider.pointAt(value));
                    require(slider.value() == value, "click must match the painted value in each orientation and direction");
                }
            }
        }
    }
    slider.setOrientation(Qt::Horizontal);
    slider.setLayoutDirection(Qt::LeftToRight);
    slider.setInvertedAppearance(false);
    slider.resize(420, 40);
}

void settingScales(Slider& slider)
{
    const auto& passes = setting("passes");
    slider.setRange(static_cast<int>(passes.minimum), static_cast<int>(passes.maximum));
    QTest::mouseClick(&slider, Qt::LeftButton, Qt::NoModifier, slider.pointAt(3));
    require(dlsslop_gui::sliderValue(passes, slider.value()) == 3, "integer setting click must remain integral");

    // Give the continuous slider a span divisible by four for exact quarter
    // positions; the style, including its focused handle width, owns geometry.
    slider.setRange(0, 10000);
    const int span = slider.pointAt(10000).x() - slider.pointAt(0).x();
    slider.resize(slider.width() + (4 - span % 4) % 4, slider.height());
    const auto& linear = setting("working-scale");
    QTest::mouseClick(&slider, Qt::LeftButton, Qt::NoModifier, slider.pointAt(2500));
    require(slider.value() == 2500 && dlsslop_gui::sliderValue(linear, slider.value()) == 0.6875,
            "linear setting click must preserve the selected fractional position");
    const auto& logarithmic = setting("white-point-scale");
    QTest::mouseClick(&slider, Qt::LeftButton, Qt::NoModifier, slider.pointAt(5000));
    require(slider.value() == 5000 && std::fabs(dlsslop_gui::sliderValue(logarithmic, slider.value()) - 1) < 1e-12,
            "logarithmic midpoint must select the geometric midpoint");
    require(dlsslop_gui::sliderPosition(logarithmic, 1) == 5000, "numeric log entry must match the clicked handle");
    for (const int value : {0, 10000}) {
        QTest::mouseClick(&slider, Qt::LeftButton, Qt::NoModifier, slider.pointAt(value));
        require(slider.value() == value, "continuous slider must reach both exact endpoints");
    }
}

} // namespace

int main(int argc, char** argv)
{
    QApplication application(argc, argv);
    QApplication::setStyle(QStyleFactory::create("Fusion"));
    application.setStyleSheet(controllerTheme());
    QFrame card;
    card.setObjectName("card");
    card.resize(500, 500);
    Slider slider(Qt::Horizontal, &card);
    slider.setGeometry(30, 30, 420, 40);
    card.show();
    slider.setFocus();
    QApplication::processEvents();
    try {
        interaction(slider);
        directions(slider);
        settingScales(slider);
        std::puts("GUI slider tests passed: absolute press, drag, release, tracking, keyboard, orientation, RTL and setting scales");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
