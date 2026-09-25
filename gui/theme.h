// SPDX-License-Identifier: MIT
#pragma once
#include <QString>

// Qt widget styling only: this is compiled into the executable and interpreted
// by Qt's widget style engine. It contains no HTML, JavaScript, QML or web view.
inline QString controllerTheme()
{
    return QStringLiteral(R"(
QWidget { background: #111820; color: #e5edf4; font-size: 14px; }
QWidget#page, QScrollArea, QScrollArea > QWidget > QWidget { background: #111820; border: none; }
QLabel { background: transparent; }
QLabel#title { font-size: 27px; font-weight: 600; }
QLabel#subtitle, QLabel#description { color: #9caebd; }
QLabel#sectionTitle { font-size: 22px; font-weight: 600; }
QLabel#settingTitle { font-weight: 600; }
QFrame#card { background: #1b2530; border-radius: 14px; }
QFrame#card QWidget { background: transparent; }
QFrame#card QDoubleSpinBox, QFrame#card QSpinBox,
QFrame#card QComboBox { background: #101a24; }
QListWidget { background: #111820; border: none; outline: none; }
QListWidget::item { padding: 13px 16px; margin: 3px 0; border-radius: 9px; color: #a7b7c5; }
QListWidget::item:selected { background: #203c42; color: #8ee4d0; }
QListWidget::item:hover { background: #25323f; }
QPushButton { background: #263644; border: 1px solid transparent; border-radius: 8px; padding: 9px 14px; }
QPushButton:hover { background: #324858; }
QPushButton:pressed { background: #1d4c50; }
QPushButton:focus, QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus,
QComboBox:focus { border: 1px solid #78d8c1; }
QPushButton#primary { background: #83deca; color: #10282b; font-weight: 600; }
QPushButton#primary:hover { background: #a1ebdb; }
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox { background: #1b2530; border: 1px solid #354553;
    border-radius: 7px; padding: 7px; selection-background-color: #2e6b69; }
QDoubleSpinBox, QSpinBox { min-width: 105px; }
QComboBox::drop-down { border: none; width: 25px; }
QSlider, QFrame#card QSlider { background: transparent; border: none; outline: none; }
QSlider::groove:horizontal { height: 5px; background: #3a4b59; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #7ad9c1; border-radius: 2px; }
QSlider::handle:horizontal { background: #9ce9d6; width: 16px; margin: -6px 0; border-radius: 8px; }
QSlider::handle:horizontal:focus { background: #ffffff; }
QCheckBox { spacing: 10px; }
QCheckBox::indicator { width: 20px; height: 20px; border-radius: 6px; border: 1px solid #627685; background: #101a24; }
QCheckBox::indicator:checked { width: 12px; height: 12px; background: #82dfc8; image: none; border: 5px solid #35685f; }
QCheckBox:focus { color: #9ce9d6; }
QWidget:disabled { color: #728696; }
QSlider::sub-page:horizontal:disabled { background: #536773; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #405260; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QToolTip { background: #273947; color: #edf6fa; border: 1px solid #587482; padding: 7px; }
QPlainTextEdit { background: #141f29; border: 1px solid #354553; border-radius: 8px; }
)");
}
