#include <mutex>
#include <spdlog/cfg/helpers.h>
#include <spdlog/details/os.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <type_traits>
#include "logger/logger.h"

namespace UC::Logger {

static spdlog::level::level_enum SpdLevels[] = {spdlog::level::debug, spdlog::level::info,
                                                spdlog::level::warn, spdlog::level::err};

class SpdLogger : public ILogger {
    std::shared_ptr<spdlog::logger> logger_;
    std::mutex mutex_;

public:
    SpdLogger() : logger_{nullptr} {}

protected:
    void Log(Level&& lv, SourceLocation&& loc, std::string&& msg) override
    {
        auto logger = this->Make();
        auto level = SpdLevels[static_cast<int>(lv)];
        logger->log(spdlog::source_loc{loc.file, loc.line, loc.func}, level, std::move(msg));
    }

private:
    std::shared_ptr<spdlog::logger> Make()
    {
        if (this->logger_) { return this->logger_; }
        std::lock_guard<std::mutex> lg(this->mutex_);
        if (this->logger_) { return this->logger_; }
        const std::string name = "UC";
        const std::string envLevel = name + "_LOGGER_LEVEL";
        try {
            this->logger_ = spdlog::stdout_color_mt(name);
            this->logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%f][%n][%^%L%$] %v [%P,%t][%s:%#,%!]");
            auto level = spdlog::details::os::getenv(envLevel.c_str());
            if (!level.empty()) { spdlog::cfg::helpers::load_levels(level); }
            return this->logger_;
        } catch (...) {
            return spdlog::default_logger();
        }
    }
};

ILogger* Make()
{
    static SpdLogger logger;
    return &logger;
}

} // namespace UC::Logger
