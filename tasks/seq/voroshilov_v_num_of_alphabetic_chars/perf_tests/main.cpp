#include <gtest/gtest.h>

#include <vector>
#include <string>

#include "core/perf/include/perf.hpp"
#include "seq/voroshilov_v_num_of_alphabetic_chars/include/ops_seq.hpp"

TEST(voroshilov_v_num_of_alphabetic_chars_seq_perf, test_pipeline_run_seq) {
    std::string str_0(10000, '0');
    std::string str_1(10000, '1');
    std::string str_2(10000, '2');
    std::string str_plus(10000, '+');
    std::string str_a(10000, 'a');
    std::string str_B(10000, 'A');
    std::string str_y(10000, 'y');
    std::string str_Z(10000, 'Z');
    std::string str = str_0 + str_1 + str_2 + str_plus + str_a + str_B + str_y + str_Z;

    int initial_num = 0;
    int expected_num = 40000;

    // Create data
    std::vector<char> in(str.length());
    for (size_t i = 0; i < in.size(); i++)
        in[i] = str[i];
    std::vector<int> out(1, initial_num);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());

    // Create Task
    auto alphabetCharsTaskSequential = std::make_shared<voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential>(taskDataSeq);

    // Create Perf attributes
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 1000;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perfAttr->current_timer = [&] {
        auto current_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
        return static_cast<double>(duration) * 1e-9;
    };

    // Create and init perf results
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Create Perf analyzer
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(alphabetCharsTaskSequential);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(expected_num, out[0]);
}

TEST(voroshilov_v_num_of_alphabetic_chars_seq_perf, test_task_run_seq) {
    std::string str_0(10000, '0');
    std::string str_1(10000, '1');
    std::string str_2(10000, '2');
    std::string str_plus(10000, '+');
    std::string str_a(10000, 'a');
    std::string str_B(10000, 'A');
    std::string str_y(10000, 'y');
    std::string str_Z(10000, 'Z');
    std::string str = str_0 + str_1 + str_2 + str_plus + str_a + str_B + str_y + str_Z;

    int initial_num = 0;
    int expected_num = 40000;

    // Create data
    std::vector<char> in(str.length());
    for (size_t i = 0; i < in.size(); i++)
        in[i] = str[i];
    std::vector<int> out(1, initial_num);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());

    // Create Task
    auto alphabetCharsTaskSequential = std::make_shared<voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential>(taskDataSeq);

    // Create Perf attributes
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 1000;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perfAttr->current_timer = [&] {
        auto current_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
        return static_cast<double>(duration) * 1e-9;
    };

    // Create and init perf results
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Create Perf analyzer
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(alphabetCharsTaskSequential);
    perfAnalyzer->task_run(perfAttr, perfResults);
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(expected_num, out[0]);
}
