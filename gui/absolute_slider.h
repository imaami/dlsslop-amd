// SPDX-License-Identifier: MIT
#pragma once
#include <QMouseEvent>
#include <QSlider>
#include <QStyle>
#include <QStyleOptionSlider>
#include <algorithm>

namespace dlsslop_gui {

// Left clicks place the handle at the pointer instead of taking a page step.
// Keep QSlider's range, tracking, signals, keyboard and wheel behavior.
class AbsoluteSlider : public QSlider {
    bool dragging_ = false;

    int positionValue(const QPoint& point) const
    {
        QStyleOptionSlider option;
        initStyleOption(&option);
        const QRect groove = style()->subControlRect(QStyle::CC_Slider, &option,
                                                     QStyle::SC_SliderGroove, this);
        const QRect handle = style()->subControlRect(QStyle::CC_Slider, &option,
                                                     QStyle::SC_SliderHandle, this);
        const bool horizontal = orientation() == Qt::Horizontal;
        const int length = horizontal ? handle.width() : handle.height();
        const int start = horizontal ? groove.left() : groove.top();
        const int span = std::max(0, (horizontal ? groove.width() : groove.height()) - length);
        // QRect's integer center rounds down for an even-sized handle.
        const int pixel = (horizontal ? point.x() : point.y()) - start - (length - 1) / 2;
        return QStyle::sliderValueFromPosition(minimum(), maximum(), pixel, span, option.upsideDown);
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
