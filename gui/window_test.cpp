// SPDX-License-Identifier: MIT
// Drives the real controller window over a fixture channel on an offscreen
// display, with input delivered through the window system like a real mouse.
#define main dlsslop_gui_main
#include "main.cpp"
#undef main
#include <QScrollBar>
#include <QTest>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

bool editor(const QWidget* widget)
{
    return widget->isVisible() && widget->isEnabled() && !qobject_cast<const QScrollBar*>(widget) &&
           (qobject_cast<const QAbstractSlider*>(widget) || qobject_cast<const QAbstractSpinBox*>(widget) ||
            qobject_cast<const QComboBox*>(widget));
}

QVariant state(const QWidget* widget)
{
    return widget->property(qobject_cast<const QComboBox*>(widget) ? "currentIndex" : "value");
}

QScrollArea* shownPage(QWidget& window)
{
    return static_cast<QScrollArea*>(window.findChild<QStackedWidget*>()->currentWidget());
}

// Turns the wheel one notch over the widget's centre; a positive notch turns up.
void wheel(QWidget& window, QWidget* widget, int notch)
{
    QTest::wheelEvent(window.windowHandle(), widget->mapTo(&window, widget->rect().center()), QPoint(0, notch));
    QApplication::processEvents();
}

// Scrolls the shown page to the widget and returns the page's position.
int reveal(QWidget& window, QWidget* widget)
{
    auto* area = shownPage(window);
    area->ensureWidgetVisible(widget);
    QApplication::processEvents();
    return area->verticalScrollBar()->value();
}

// A notch that moves the control away from the end it rests at.
int notchTowardsChange(const QWidget* widget)
{
    if (const auto* combo = qobject_cast<const QComboBox*>(widget)) return combo->currentIndex() > 0 ? 120 : -120;
    return state(widget).toDouble() < widget->property("maximum").toDouble() ? 120 : -120;
}

template <class Widget> Widget* named(QWidget& window, const QString& name)
{
    for (auto* widget : window.findChildren<Widget*>())
        if (widget->accessibleName() == name) return widget;
    throw std::runtime_error("missing test widget");
}

void unfocusedEditorsScrollThePage(QWidget& window, const ShmHeader& header)
{
    const uint32_t control = header.controlSeq.load();
    auto* stack = window.findChild<QStackedWidget*>();
    // Keep focus outside the pages, so switching pages moves no focus.
    window.findChild<QListWidget*>()->setFocus();
    int editors = 0, scrolled = 0;
    for (int page = 0; page < stack->count(); ++page) {
        stack->setCurrentIndex(page);
        QApplication::processEvents();
        const auto* bar = shownPage(window)->verticalScrollBar();
        for (auto* widget : shownPage(window)->widget()->findChildren<QWidget*>()) {
            if (!editor(widget)) continue;
            ++editors;
            const QVariant before = state(widget);
            const int position = reveal(window, widget);
            wheel(window, widget, position < bar->maximum() ? -120 : 120);
            require(state(widget) == before, "the wheel edited an unfocused control");
            require(!widget->hasFocus(), "the wheel focused a control");
            if (bar->maximum() == 0) continue;
            require(bar->value() != position, "the wheel over an unfocused control did not scroll its page");
            ++scrolled;
        }
    }
    require(editors > 40 && scrolled > 0, "the wheel test reached too few controls");
    QTest::qWait(100); // Longer than the write throttle.
    require(header.controlSeq.load() == control, "the wheel over unfocused controls wrote the channel");
}

void focusedEditorsTakeTheWheel(QWidget& window, const ShmHeader& header)
{
    auto* stack = window.findChild<QStackedWidget*>();
    auto* navigation = window.findChild<QListWidget*>();
    const uint32_t control = header.controlSeq.load();
    const struct { int page; QWidget* widget; } cases[] = {
        {0, named<QSlider>(window, "Intensity slider")},
        {0, named<QDoubleSpinBox>(window, "Local tone")},
        {1, named<QComboBox>(window, "Transfer")},
    };
    for (const auto& c : cases) {
        navigation->setFocus();
        stack->setCurrentIndex(c.page);
        c.widget->setFocus();
        QApplication::processEvents();
        require(c.widget->hasFocus(), "test control did not take focus");
        const QVariant before = state(c.widget);
        const int position = reveal(window, c.widget);
        wheel(window, c.widget, notchTowardsChange(c.widget));
        require(state(c.widget) != before, "the wheel did not edit a focused control");
        require(shownPage(window)->verticalScrollBar()->value() == position,
                "the wheel over a focused control scrolled its page");
    }
    require(QTest::qWaitFor([&] { return header.controlSeq.load() != control; }, 1000),
            "a wheel edit was not written to the channel");
}

void editsWriteAtOnceThenCoalesce(QWidget& window, const ShmHeader& header)
{
    auto* hold = named<QCheckBox>(window, "Hold");
    auto* intensity = named<QDoubleSpinBox>(window, "Intensity");
    QTest::qWait(200); // Let any earlier write windows close.
    const uint32_t control = header.controlSeq.load();
    hold->toggle();
    require(header.controlSeq.load() == control + 1 && header.holdFrame.load() == hold->isChecked(),
            "an edit was not written at once");
    intensity->setValue(0.5);
    intensity->setValue(0.25);
    hold->toggle();
    require(header.controlSeq.load() == control + 1, "edits within the write window were not coalesced");
    require(QTest::qWaitFor([&] { return header.controlSeq.load() != control + 1; }, 1000),
            "coalesced edits were never written");
    require(header.controlSeq.load() == control + 2 && BitsToFloat(header.intensityBits.load()) == 0.25f &&
            header.holdFrame.load() == hold->isChecked(), "coalesced edits were not written as one final batch");
    QTest::qWait(100);
    require(header.controlSeq.load() == control + 2, "an idle write window wrote again");
    hold->toggle();
    require(header.controlSeq.load() == control + 3, "an edit after an idle window was not written at once");
}

// Answers the controller's modal warnings, recording the last one's title.
class WarningCloser : public QObject {
public:
    QString title;

    WarningCloser() { startTimer(10); }

protected:
    void timerEvent(QTimerEvent*) override
    {
        auto* box = qobject_cast<QMessageBox*>(QApplication::activeModalWidget());
        if (!box) return;
        title = box->windowTitle();
        box->done(QMessageBox::Ok);
    }
};

// Shows the window with the text typed, not yet committed, into the focused
// Intensity field, and no write window open.
void typeIntensity(QWidget& window, const QString& text)
{
    window.show();
    window.activateWindow();
    require(QTest::qWaitForWindowActive(&window), "window not active");
    window.findChild<QStackedWidget*>()->setCurrentIndex(0);
    auto* intensity = named<QDoubleSpinBox>(window, "Intensity");
    intensity->setFocus();
    QApplication::processEvents();
    require(intensity->hasFocus(), "test control did not take focus");
    intensity->selectAll();
    QTest::keyClicks(intensity, text);
    require(intensity->text() == text && intensity->value() != text.toDouble(), "typed text was committed early");
    QTest::qWait(200); // Let any earlier write windows close.
}

// Closing commits typed text, which writes at once when no write window is
// open. If that write finds the channel gone, closing must still warn.
void closingSendsTypedTextOrWarns(QWidget& window, const ShmHeader& header, const std::string& path)
{
    WarningCloser warnings;
    const uint32_t control = header.controlSeq.load();
    typeIntensity(window, "0.3");
    window.close();
    require(warnings.title.isEmpty(), "closing over a healthy channel warned");
    require(header.controlSeq.load() == control + 1 && BitsToFloat(header.intensityBits.load()) == 0.3f,
            "closing did not send the typed value");
    typeIntensity(window, "0.4");
    require(!unlink(path.c_str()), "unlink fixture");
    window.close();
    require(warnings.title == "Last change was not sent", "closing lost a typed change without a warning");
    require(header.controlSeq.load() == control + 1, "a write reached the unlinked channel");
}

} // namespace

int main(int argc, char** argv)
{
    char directory[] = "/tmp/dlsslop-amd-gui-window-test-XXXXXX";
    if (!mkdtemp(directory)) return 1;
    const std::string path = std::string(directory) + "/channel";
    int fd = -1;
    ShmHeader* header = nullptr;
    int result = 0;
    try {
        fd = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        require(fd >= 0, "create fixture");
        require(!ftruncate(fd, static_cast<off_t>(ShmTotalBytes())), "size fixture");
        void* memory = mmap(nullptr, kHeaderBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        require(memory != MAP_FAILED, "map fixture");
        header = static_cast<ShmHeader*>(memory);
        ShmInitNativeDefaults(header);
        QApplication application(argc, argv);
        QApplication::setStyle(QStyleFactory::create("Fusion"));
        QLocale::setDefault(QLocale::c());
        application.setStyleSheet(controllerTheme());
        Window window(QString::fromStdString(path));
        window.show();
        require(QTest::qWaitForWindowExposed(&window), "window not exposed");
        unfocusedEditorsScrollThePage(window, *header);
        focusedEditorsTakeTheWheel(window, *header);
        editsWriteAtOnceThenCoalesce(window, *header);
        closingSendsTypedTextOrWarns(window, *header, path);
        std::puts("GUI window tests passed: page scrolling over unfocused controls, wheel edits of focused ones, "
                  "edits written at once then coalesced, and typed text sent or warned about on close");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        result = 1;
    }
    if (header) munmap(header, kHeaderBytes);
    if (fd >= 0) close(fd);
    unlink(path.c_str());
    rmdir(directory);
    return result;
}
