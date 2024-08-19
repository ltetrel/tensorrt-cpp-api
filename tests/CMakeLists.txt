include(CTest)

execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Current git commit ${GIT_COMMIT_HASH}.")

#  Downloading test data
include(ExternalData)
set(ExternalData_URL_TEMPLATES "https://data.kitware.com/api/v1/file/hashsum/%(algo)/%(hash)/download")
ExternalData_Expand_Arguments(testDetectorData
    testDetectorDataRealPath
    DATA{data/input/HOLO21_STN079_frame000990.png}
    DATA{data/input/models/darknet/darknet.onnx}
    DATA{data/input/models/darknet/detector_config.yaml}
    DATA{data/baseline/darknet/HOLO21_STN079_frame000990.txt}
)
ExternalData_Add_Target(testDetectorData)

# Configure test source
set(DATA_TESTING_DIR "data")
message(STATUS "Test data available at: " ${CMAKE_BINARY_DIR}/tests/data)
configure_file(testDetector.cpp.in testDetector.cpp @ONLY)

# Configure test build
add_executable(testDetector ${CMAKE_CURRENT_BINARY_DIR}/testDetector.cpp)
add_dependencies(testDetector testDetectorData)
target_include_directories(testDetector PRIVATE "${PROJECT_SOURCE_DIR}/src" ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(testDetector tensorrt_cpp_api tensorrt_detector)
add_test(NAME testDarknet COMMAND testDetector "darknet")
add_test(NAME testNetharn COMMAND testDetector "netharn")
