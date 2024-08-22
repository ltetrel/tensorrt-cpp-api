#pragma once

#include <filesystem>
#include <fstream>
#include <chrono>
#include <atomic>
#include <limits>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>


namespace Utils{

inline std::vector<std::string> getFilesInDirectory(const std::string& dirPath){
    std::vector<std::string> filepaths;

    for (const auto& entry: std::filesystem::directory_iterator(dirPath)) {
        filepaths.emplace_back(entry.path().string());
    }

    return filepaths;
}

inline std::string getDirPath(const std::string& filePath){
        std::filesystem::path p = filePath;
        std::string parentPath = p.parent_path();

        return parentPath;
}

// Parser utils
template <typename T>
inline T parseFileNodeValue(cv::FileNode ocvFn){
    T value;
    
    if constexpr(std::is_same<T, bool>::value){
        value = (ocvFn.string() == "true") ? true : false;
    }
    else{
        value = ocvFn;
    }

    return value;
}

template <typename T>
inline std::vector<T> parseFileNodeVector(cv::FileNode ocvFn){
    std::vector<T> values;

    for (cv::FileNodeIterator it = ocvFn.begin(); it != ocvFn.end(); ++it){
        values.emplace_back(static_cast<T>(*it));
    }

    return values;
}

template<typename T, int N>
inline cv::Vec<T, N> parseFileNodeCVVec(cv::FileNode ocvFn){
    cv::Vec<T, N> cvVec;

    std::vector<T> vec = parseFileNodeVector<T>(ocvFn);
    cvVec = cv::Vec<T, N>(vec.data());

    return cvVec;
}

template <typename T>
inline T getValueFromMapKey(const std::unordered_map<std::string, T>& mapperTable, const std::string key){
    T value;
    
    auto it = mapperTable.find(key);
    if (it != mapperTable.end()) {
        value = it->second;
    }
    else{
        throw std::runtime_error("Cannot find key: " + key);
    }

    return value;
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
const std::string API_VERSION = "0.1";
const int MAX_INT = std::numeric_limits<int>::max();
const float MAX_FLOAT = std::numeric_limits<float>::max();

}
