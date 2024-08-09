include(CTest)

set(DATA_TESTING_DIR "${CMAKE_SOURCE_DIR}/tests/data")
#TODO put test data on girder: https://github.com/girder/girder/blob/master/tests/CMakeLists.txt
# FetchContent_Declare(test_data
#     URL https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg
#     URL_HASH SHA256=12794390cce7d0682ffc783c785e4282305684431b30b29ed75c224da24035b4
#     DOWNLOAD_NO_EXTRACT true
# )
# FetchContent_MakeAvailable(test_data)
message(STATUS "Setting test data directory to: ${DATA_TESTING_DIR}.")

configure_file(testDetector.cpp.in testDetector.cpp @ONLY)
add_executable(testDetector ${CMAKE_CURRENT_BINARY_DIR}/testDetector.cpp)
target_include_directories(testDetector PRIVATE "${PROJECT_SOURCE_DIR}/src" ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(testDetector tensorrt_cpp_api tensorrt_detector)
add_test(NAME testDarknet COMMAND testDetector "darknet")
add_test(NAME testNetharn COMMAND testDetector "netharn")
