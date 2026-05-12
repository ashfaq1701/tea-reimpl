add_executable(tea_smoke_test smoke_test.cpp)
target_link_libraries(tea_smoke_test PRIVATE tea tea_compile_flags GTest::gtest_main)
add_test(NAME tea_smoke_test COMMAND tea_smoke_test)

add_executable(test_graph test_graph.cpp)
target_link_libraries(test_graph PRIVATE tea tea_compile_flags GTest::gtest_main)
add_test(NAME test_graph COMMAND test_graph)

add_executable(test_alias test_alias.cpp)
target_link_libraries(test_alias PRIVATE tea tea_compile_flags GTest::gtest_main)
add_test(NAME test_alias COMMAND test_alias)

add_executable(test_csv_reader test_csv_reader.cpp)
target_link_libraries(test_csv_reader PRIVATE tea tea_compile_flags GTest::gtest_main)
add_test(NAME test_csv_reader COMMAND test_csv_reader)

add_executable(test_bias test_bias.cpp)
target_link_libraries(test_bias PRIVATE tea tea_compile_flags GTest::gtest_main)
add_test(NAME test_bias COMMAND test_bias)

add_executable(test_pat test_pat.cpp)
target_link_libraries(test_pat PRIVATE tea tea_compile_flags GTest::gtest_main)
add_test(NAME test_pat COMMAND test_pat)

add_executable(test_sampler test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE tea tea_compile_flags GTest::gtest_main)
add_test(NAME test_sampler COMMAND test_sampler)

add_executable(test_hpat test_hpat.cpp)
target_link_libraries(test_hpat PRIVATE tea tea_compile_flags GTest::gtest_main)
add_test(NAME test_hpat COMMAND test_hpat)

add_executable(test_node2vec test_node2vec.cpp)
target_link_libraries(test_node2vec PRIVATE tea tea_compile_flags GTest::gtest_main)
add_test(NAME test_node2vec COMMAND test_node2vec)

add_executable(test_walk_engine test_walk_engine.cpp)
target_link_libraries(test_walk_engine PRIVATE tea tea_compile_flags GTest::gtest_main)
add_test(NAME test_walk_engine COMMAND test_walk_engine)

add_executable(test_aux_index test_aux_index.cpp)
target_link_libraries(test_aux_index PRIVATE tea tea_compile_flags GTest::gtest_main)
add_test(NAME test_aux_index COMMAND test_aux_index)

add_executable(test_solo test_solo.cpp)
target_link_libraries(test_solo PRIVATE tea tea_compile_flags GTest::gtest_main)
add_test(NAME test_solo COMMAND test_solo)

add_executable(test_temporal_start test_temporal_start.cpp)
target_link_libraries(test_temporal_start PRIVATE tea tea_compile_flags GTest::gtest_main)
add_test(NAME test_temporal_start COMMAND test_temporal_start)
