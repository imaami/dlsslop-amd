// SPDX-License-Identifier: MIT
#include "channel.h"
#include "absolute_slider.h"
#include "slider_scale.h"
#include "theme.h"
#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QLocale>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QStackedWidget>
#include <QStyleFactory>
#include <QTimer>
#include <QVBoxLayout>
#include <algorithm>
#include <cstdio>
#include <functional>
#include <getopt.h>
#include <vector>

namespace {
using dlsslop_control::kSettings;
using dlsslop_gui::Channel;

constexpr double kArrowSteps[] = {0.001, 0.01, 0.1};
constexpr int kDefaultArrowStep = 1;

QString title(QString name)
{
    name.replace('-', ' ');
    name[0] = name[0].toUpper();
    return name;
}

struct Editor {
    QWidget* root{};
    QDoubleSpinBox* number{};
    QSlider* slider{};
    QCheckBox* checkbox{};
    QComboBox* combo{};
    QPushButton* reset{};
    double defaultValue{};
};

class Window final : public QWidget {
    QLineEdit* path_{};
    QLabel* state_{};
    QWidget* controls_{};
    QPlainTextEdit* status_{};
    QTimer throttle_;
    std::vector<Editor> editors_;
    std::map<std::size_t, double> pending_;
    bool connected_ = false;
    dev_t device_{};
    ino_t inode_{};
    std::string activePath_;
    double step_ = kArrowSteps[kDefaultArrowStep];

    bool sameChannel(const Channel& channel) const
    { return connected_ && channel.device() == device_ && channel.inode() == inode_; }

    void failed(const QString& error)
    {
        throttle_.stop();
        pending_.clear();
        connected_ = false;
        controls_->setEnabled(false);
        state_->setText("Disconnected — " + error + ". Start the worker, then Connect / refresh.");
    }

    bool flush()
    {
        throttle_.stop();
        if (pending_.empty()) return true;
        try {
            Channel channel(activePath_, true);
            if (!sameChannel(channel)) throw std::runtime_error("Channel replaced; refresh before editing");
            channel.write(pending_);
            pending_.clear();
            state_->setText("Settings sent • Closing this window leaves the worker running");
            return true;
        } catch (const std::exception& error) {
            failed(QString::fromUtf8(error.what()));
            return false;
        }
    }

    void queue(std::size_t index, double value)
    {
        if (!connected_) return;
        pending_[index] = value;
        // Write at once, then coalesce the next 40 ms of edits into one write.
        if (!throttle_.isActive() && flush()) throttle_.start(40);
    }

    // Page scrolling must not edit a control the pointer happens to cross.
    bool eventFilter(QObject* object, QEvent* event) override
    {
        if (event->type() != QEvent::Wheel || static_cast<QWidget*>(object)->hasFocus()) return false;
        event->ignore(); // Unaccepted, it propagates to the page's scroll area.
        return true;
    }

    void wheelNeedsFocus(QWidget* editor)
    {
        editor->setFocusPolicy(Qt::StrongFocus); // WheelFocus would take focus before the filter runs.
        editor->installEventFilter(this);
    }

    void display(std::size_t index, double value)
    {
        auto& e = editors_[index];
        if (e.number) { const QSignalBlocker blocked(e.number); e.number->setValue(value); }
        if (e.slider) { const QSignalBlocker blocked(e.slider); e.slider->setValue(dlsslop_gui::sliderPosition(kSettings[index], value)); }
        if (e.checkbox) { const QSignalBlocker blocked(e.checkbox); e.checkbox->setChecked(value != 0); }
        if (e.combo) { const QSignalBlocker blocked(e.combo); e.combo->setCurrentIndex(e.combo->findData(static_cast<int>(value))); }
    }

    void refresh()
    {
        if (!flush()) return;
        try {
            const auto path = path_->text().toLocal8Bit();
            Channel channel(path.constData(), false);
            auto* h = channel.header();
            ShmHeader defaults{};
            ShmInitNativeDefaults(&defaults, dlsslop_control::workerBypass(h));
            // Reject invalid live values rather than displaying a silently clamped setting.
            std::vector<double> values;
            for (const auto& s : kSettings) {
                const double value = dlsslop_control::value(s, (h->*s.field).load());
                if (!dlsslop_control::inRange(s, value))
                    throw std::runtime_error(std::string("Invalid live setting: ") + s.name);
                values.push_back(value);
            }
            for (std::size_t i = 0; i < editors_.size(); ++i) {
                auto& e = editors_[i];
                e.defaultValue = dlsslop_control::value(kSettings[i], (defaults.*kSettings[i].field).load());
                e.reset->setToolTip(QString("Reset to %1").arg(e.defaultValue, 0, 'g', 9));
                display(i, values[i]);
            }
            device_ = channel.device();
            inode_ = channel.inode();
            activePath_ = path.constData();
            connected_ = true;
            controls_->setEnabled(true);
            const auto reason = ShmLoadString(h->helperReasonSeq, h->helperReason, kReasonBytes);
            status_->setPlainText(QString("Snapshot on refresh\n\nProtocol: %1\nWorker state: %2\nModel up: %3\nStop requested: %4\n"
                "Request / response: %5 / %6\nProxy: %7 × %8\nNeural raster limit: %9 × %10\n"
                "Network: %11 ms\nUpload: %12 ms\nReadback: %13 ms\nReason: %14")
                .arg(h->version.load()).arg(h->helperState.load()).arg(h->modelUp.load()).arg(h->quit.load())
                .arg(h->seq_req.load()).arg(h->seq_resp.load()).arg(h->width.load()).arg(h->height.load())
                .arg(h->nativeModelMaxWidth.load()).arg(h->nativeModelMaxHeight.load())
                .arg(BitsToFloat(h->helperEvalMsBits.load())).arg(BitsToFloat(h->helperUploadMsBits.load()))
                .arg(BitsToFloat(h->helperReadbackMsBits.load())).arg(QString::fromStdString(reason)));
            state_->setText("Channel connected • Changes apply live • Refresh to read changes made elsewhere");
        } catch (const std::exception& error) { failed(QString::fromUtf8(error.what())); }
    }

    void action(const std::function<void(Channel&)>& operation, const QString& success)
    {
        if (!flush()) return;
        try {
            Channel channel(activePath_, true);
            if (!sameChannel(channel)) throw std::runtime_error("Channel replaced; refresh before editing");
            operation(channel);
            refresh();
            if (connected_) state_->setText(success);
        } catch (const std::exception& error) { failed(QString::fromUtf8(error.what())); }
    }

    QWidget* makeEditor(std::size_t index)
    {
        const auto& s = kSettings[index];
        auto& e = editors_[index];
        auto* card = new QFrame;
        card->setObjectName("card");
        e.root = card;
        card->setToolTip(QString("-%1 / --%2\n%3\nRange: %4…%5")
            .arg(QChar(s.shortName)).arg(s.name).arg(s.help).arg(s.minimum).arg(s.maximum));
        auto* layout = new QVBoxLayout(card);
        layout->setContentsMargins(18, 14, 18, 14);
        layout->setSpacing(9);
        auto* row = new QHBoxLayout;
        auto* label = new QLabel(title(s.name));
        label->setObjectName("settingTitle");
        row->addWidget(label);
        row->addStretch();
        if (s.choices) {
            const auto modes = QString(s.choices).split('|');
            e.combo = new QComboBox;
            e.combo->setMinimumWidth(190);
            for (int i = 0; i < modes.size(); ++i) e.combo->addItem(modes[i], static_cast<int>(s.minimum) + i);
            e.combo->setAccessibleName(title(s.name));
            wheelNeedsFocus(e.combo);
            row->addWidget(e.combo);
            connect(e.combo, &QComboBox::currentIndexChanged, this, [this, index](int) {
                queue(index, editors_[index].combo->currentData().toInt());
            });
        } else if (!s.isFloat && s.maximum == 1) {
            e.checkbox = new QCheckBox("On");
            e.checkbox->setAccessibleName(title(s.name));
            row->addWidget(e.checkbox);
            connect(e.checkbox, &QCheckBox::toggled, this, [this, index](bool checked) { queue(index, checked ? 1 : 0); });
        } else {
            e.number = new QDoubleSpinBox;
            e.number->setDecimals(s.isFloat ? 6 : 0);
            e.number->setRange(s.minimum, s.maximum);
            e.number->setSingleStep(s.isFloat ? step_ : 1);
            e.number->setKeyboardTracking(false);
            e.number->setAccessibleName(title(s.name));
            wheelNeedsFocus(e.number);
            row->addWidget(e.number);
            connect(e.number, &QDoubleSpinBox::valueChanged, this, [this, index](double value) {
                auto* slider = editors_[index].slider;
                if (slider) { const QSignalBlocker blocked(slider); slider->setValue(dlsslop_gui::sliderPosition(kSettings[index], value)); }
                queue(index, value);
            });
        }
        e.reset = new QPushButton("Reset");
        e.reset->setAccessibleName("Reset " + title(s.name));
        row->addWidget(e.reset);
        connect(e.reset, &QPushButton::clicked, this, [this, index] {
            display(index, editors_[index].defaultValue);
            queue(index, editors_[index].defaultValue);
        });
        layout->addLayout(row);
        auto* help = new QLabel(s.help);
        help->setWordWrap(true);
        help->setObjectName("description");
        layout->addWidget(help);
        if (e.number && !dlsslop_control::fixed(s)) {
            e.slider = new dlsslop_gui::AbsoluteSlider(Qt::Horizontal);
            e.slider->setRange(s.isFloat ? 0 : static_cast<int>(s.minimum), s.isFloat ? 10000 : static_cast<int>(s.maximum));
            e.slider->setAccessibleName(title(s.name) + " slider");
            wheelNeedsFocus(e.slider);
            e.slider->setToolTip(dlsslop_gui::logarithmicSlider(s) ? "Logarithmic sweep; use the numeric field for an exact value" :
                                                   "Live sweep; use the numeric field for an exact value");
            layout->addWidget(e.slider);
            connect(e.slider, &QSlider::valueChanged, this, [this, index](int position) {
                auto* number = editors_[index].number;
                const QSignalBlocker blocked(number);
                number->setValue(dlsslop_gui::sliderValue(kSettings[index], position));
                queue(index, number->value());
            });
            connect(e.slider, &QSlider::sliderReleased, this, [this] { flush(); });
        }
        if (dlsslop_control::fixed(s)) {
            e.reset->setEnabled(false);
            if (e.number) e.number->setEnabled(false);
            if (e.checkbox) e.checkbox->setEnabled(false);
            help->setText(QString(s.help) + ". Read-only: alternate captured configurations are unavailable.");
        }
        return card;
    }

    void closeEvent(QCloseEvent* event) override
    {
        // Committing typed text may write at once, and fail there.
        const bool connected = connected_;
        for (auto& editor : editors_)
            if (editor.number && editor.number->hasFocus()) editor.number->interpretText();
        if (!flush() || connected_ != connected) {
            QMessageBox::warning(this, "Last change was not sent", "The control channel became unavailable. The final pending change was not applied.");
        }
        event->accept(); // Never stop the worker or restore settings on exit.
    }

public:
    explicit Window(const QString& path) : editors_(std::size(kSettings))
    {
        setWindowTitle("DLSSLOP AMD · Render controls");
        resize(1030, 800);
        setMinimumSize(790, 560);
        throttle_.setSingleShot(true);
        connect(&throttle_, &QTimer::timeout, this, [this] { if (!pending_.empty() && flush()) throttle_.start(40); });
        auto* outer = new QVBoxLayout(this);
        outer->setContentsMargins(28, 24, 28, 18);
        outer->setSpacing(14);
        auto* heading = new QLabel("Render controls");
        heading->setObjectName("title");
        outer->addWidget(heading);
        auto* subtitle = new QLabel("Explore the image. Keep the details.");
        subtitle->setObjectName("subtitle");
        outer->addWidget(subtitle);
        auto* connection = new QHBoxLayout;
        path_ = new QLineEdit(path);
        path_->setAccessibleName("Shared-memory control channel");
        connection->addWidget(path_, 1);
        auto* refreshButton = new QPushButton("Connect / refresh");
        refreshButton->setObjectName("primary");
        connection->addWidget(refreshButton);
        outer->addLayout(connection);
        connect(refreshButton, &QPushButton::clicked, this, [this] { refresh(); });
        connect(path_, &QLineEdit::returnPressed, this, [this] { refresh(); });
        connect(path_, &QLineEdit::textEdited, this, [this] {
            flush();
            connected_ = false;
            controls_->setEnabled(false);
            state_->setText("Channel path changed — Connect / refresh to use it");
        });
        controls_ = new QWidget;
        auto* body = new QHBoxLayout(controls_);
        body->setContentsMargins(0, 0, 0, 0);
        body->setSpacing(22);
        auto* navigation = new QListWidget;
        navigation->setFixedWidth(180);
        navigation->setAccessibleName("Control sections");
        auto* stack = new QStackedWidget;
        body->addWidget(navigation);
        body->addWidget(stack, 1);
        const QStringList names{"Neural passes", "Composition", "Image & HDR", "Motion", "Compare & inspect", "Model configuration", "Actions & status"};
        std::vector<QVBoxLayout*> pages;
        for (const auto& name : names) {
            navigation->addItem(name);
            auto* page = new QWidget;
            page->setObjectName("page");
            auto* layout = new QVBoxLayout(page);
            layout->setContentsMargins(0, 0, 12, 0);
            layout->setSpacing(12);
            auto* section = new QLabel(name);
            section->setObjectName("sectionTitle");
            layout->addWidget(section);
            auto* scroll = new QScrollArea;
            scroll->setWidgetResizable(true);
            scroll->setFrameShape(QFrame::NoFrame);
            scroll->setWidget(page);
            stack->addWidget(scroll);
            pages.push_back(layout);
        }
        for (std::size_t i = 0; i < editors_.size(); ++i) pages[kSettings[i].section]->addWidget(makeEditor(i));
        auto* precisionRow = new QHBoxLayout;
        precisionRow->addWidget(new QLabel("Numeric arrow step"));
        auto* step = new QComboBox;
        for (double value : kArrowSteps) step->addItem(QString::number(value), value);
        step->setCurrentIndex(kDefaultArrowStep);
        wheelNeedsFocus(step);
        precisionRow->addWidget(step);
        precisionRow->addStretch();
        pages[0]->insertLayout(1, precisionRow);
        connect(step, &QComboBox::currentIndexChanged, this, [this, step](int) {
            step_ = step->currentData().toDouble();
            for (std::size_t i = 0; i < editors_.size(); ++i)
                if (editors_[i].number && kSettings[i].isFloat) editors_[i].number->setSingleStep(step_);
        });
        auto* captureRow = new QHBoxLayout;
        auto* count = new QSpinBox;
        count->setRange(0, 64);
        count->setValue(1);
        count->setAccessibleName("Capture frame count");
        wheelNeedsFocus(count);
        captureRow->addWidget(new QLabel("Capture frames"));
        captureRow->addWidget(count);
        auto* capture = new QPushButton("Request capture");
        captureRow->addWidget(capture);
        pages[6]->addLayout(captureRow);
        connect(capture, &QPushButton::clicked, this, [this, count] {
            action([count](Channel& c) { c.capture(static_cast<unsigned>(count->value())); },
                   "Capture requested • Uses the running layer's configured capture directory");
        });
        auto* actions = new QHBoxLayout;
        auto* reset = new QPushButton("Reset all settings");
        auto* stop = new QPushButton("Request stop");
        auto* resume = new QPushButton("Clear stop request");
        actions->addWidget(reset); actions->addWidget(stop); actions->addWidget(resume);
        pages[6]->addLayout(actions);
        connect(reset, &QPushButton::clicked, this, [this] {
            if (QMessageBox::question(this, "Reset settings", "Restore all controls to their worker-mode defaults?") == QMessageBox::Yes)
                action([](Channel& c) { c.reset(); }, "Settings reset • Worker stop state was preserved");
        });
        connect(stop, &QPushButton::clicked, this, [this] {
            if (QMessageBox::question(this, "Stop processing", "Request the worker and layer to stop? Restarting the worker must be done separately.") == QMessageBox::Yes)
                action([](Channel& c) { c.stop(true); }, "Stop requested • This is a request, not confirmation of worker exit");
        });
        connect(resume, &QPushButton::clicked, this, [this] {
            action([](Channel& c) { c.stop(false); }, "Stop request cleared • An exited worker is not restarted");
        });
        auto* note = new QLabel("Status is a snapshot. Use Connect / refresh to reread settings and status.\n"
            "Worker launch options remain in the worker CLI. Closing this controller never stops it.");
        note->setWordWrap(true);
        note->setObjectName("description");
        pages[6]->addWidget(note);
        status_ = new QPlainTextEdit;
        status_->setReadOnly(true);
        status_->setMinimumHeight(280);
        pages[6]->addWidget(status_);
        for (auto* page : pages) page->addStretch();
        connect(navigation, &QListWidget::currentRowChanged, stack, &QStackedWidget::setCurrentIndex);
        navigation->setCurrentRow(0);
        outer->addWidget(controls_, 1);
        state_ = new QLabel;
        state_->setWordWrap(true);
        state_->setObjectName("subtitle");
        outer->addWidget(state_);
        refresh();
    }
};
} // namespace

int main(int argc, char** argv)
{
    std::string path = ShmNativeChannelPath();
    const option options[] = {{"shm", required_argument, nullptr, 's'}, {"help", no_argument, nullptr, 'h'}, {nullptr, 0, nullptr, 0}};
    int code;
    // Parse help before QApplication, so --help works without a display server.
    while ((code = getopt_long(argc, argv, "+s:h", options, nullptr)) != -1) {
        if (code == 'h') {
            std::printf("Usage: dlsslop-gui [OPTION]...\n"
                "  -s, --shm PATH  Existing control channel (default: nonempty DLSSNR_SHM,\n"
                "                  otherwise /tmp/dlsslop-amd-UID/shm.bin; effective: %s)\n"
                "  -h, --help      Show help (default: off)\n"
                "Standalone Qt 6 Widgets client. Does not start or stop the worker on open/close.\n"
                "Live changes: on; numeric arrow step: %g; exact entry: six decimals.\n"
                "QT_QPA_PLATFORM selects the Qt platform (default: automatic; xcb for X11).\n",
                path.c_str(), kArrowSteps[kDefaultArrowStep]);
            return 0;
        }
        if (code != 's') return 2;
        if (!*optarg) { std::fprintf(stderr, "--shm requires a nonempty path; try --help\n"); return 2; }
        path = optarg;
    }
    if (optind != argc) { std::fprintf(stderr, "Unexpected argument; try --help\n"); return 2; }
    int qtArgc = 1;
    QApplication application(qtArgc, argv);
    QApplication::setStyle(QStyleFactory::create("Fusion"));
    QLocale::setDefault(QLocale::c());
    application.setStyleSheet(controllerTheme());
    Window window(QString::fromLocal8Bit(path.c_str()));
    window.show();
    return application.exec();
}
