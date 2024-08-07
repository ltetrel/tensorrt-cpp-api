#pragma once

#include <filesystem>
#include <fstream>
#include <chrono>
#include <atomic>

namespace Utils{

std::vector<std::string> getFilesInDirectory(const std::string& dirPath){
    std::vector<std::string> filepaths;
    for (const auto& entry: std::filesystem::directory_iterator(dirPath)) {
        filepaths.emplace_back(entry.path().string());
    }
    return filepaths;
}

std::string getDirPath(const std::string& filePath){
        std::filesystem::path p = filePath;
        std::string parentPath = p.parent_path();

        return parentPath;
}

// Utility Timer
template <typename Clock = std::chrono::high_resolution_clock>
class Stopwatch
{
    typename Clock::time_point start_point;
public:
    Stopwatch() :start_point(Clock::now()){}

    // Returns elapsed time
    template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration>
    Rep elapsedTime() const {
        std::atomic_thread_fence(std::memory_order_relaxed);
        auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start_point).count();
        std::atomic_thread_fence(std::memory_order_relaxed);
        return static_cast<Rep>(counted_time);
    }
};

using preciseStopwatch = Stopwatch<>;

// TODO: use cmake .in to define version based on `git describe` as in 
// https://gitlab.kitware.com/paraview/paraview/-/blob/master/CMakeLists.txt?ref_type=heads#L180
// https://gitlab.kitware.com/paraview/paraview/-/blob/master/CMake/paraview_plugin.h.in?ref_type=heads#L65
const std::string apiVersion = "0.1";

}
