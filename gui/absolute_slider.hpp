// SPDX-License-Identifier: MIT
#pragma once
#include <QMouseEvent>
#include <QSlider>
#include <QStyle>
#include <QStyleOptionSlider>

namespace dlsslop_gui {

// Left clicks place the handle at the pointer instead of taking a page step.
// Keep QSlider's range, tracking, signals, keyboard and wheel behavior; the
// controller window passes wheel input over an unfocused slider to its page.
class AbsoluteSlider : public QSlider {
    bool dragging_ = false;

    int positionValue(const QPoint& point) const
    {
        QStyleOptionSlider option;
        initStyleOption(&option);
        // Invert the style's painting: its handle centres at the two painted
        // ends bound the travel, whatever groove insets or margins it applies.
        const int first = option.upsideDown ? option.maximum : option.minimum;
        option.sliderPosition = first;
        const QPoint start = style()->subControlRect(QStyle::CC_Slider, &option, QStyle::SC_SliderHandle, this).center();
        option.sliderPosition = option.minimum + option.maximum - first;
        QPoint pixel = point - start;
        QPoint span = style()->subControlRect(QStyle::CC_Slider, &option, QStyle::SC_SliderHandle, this).center() - start;
        if (orientation() == Qt::Vertical) { pixel = pixel.transposed(); span = span.transposed(); }
        return QStyle::sliderValueFromPosition(option.minimum, option.maximum, pixel.x(), span.x(), option.upsideDown);
    }

protected:
    void mousePressEvent(QMouseEvent* event) override
    {
        if (event->button() != Qt::LeftButton || event->buttons() != Qt::LeftButton) {
            QSlider::mousePressEvent(event);
            return;
        }
        if (focusPolicy() & Qt::ClickFocus) setFocus(Qt::MouseFocusReason);
        dragging_ = true;
        setSliderDown(true);
        setSliderPosition(positionValue(event->position().toPoint()));
        event->accept();
    }

    void mouseMoveEvent(QMouseEvent* event) override
    {
        if (!dragging_) {
            QSlider::mouseMoveEvent(event);
            return;
        }
        setSliderPosition(positionValue(event->position().toPoint()));
        event->accept();
    }

    void mouseReleaseEvent(QMouseEvent* event) override
    {
        if (!dragging_ || event->button() != Qt::LeftButton) {
            QSlider::mouseReleaseEvent(event);
            return;
        }
        setSliderPosition(positionValue(event->position().toPoint()));
        dragging_ = false;
        setSliderDown(false);
        event->accept();
    }

public:
    using QSlider::QSlider;
};

} // namespace dlsslop_gui
