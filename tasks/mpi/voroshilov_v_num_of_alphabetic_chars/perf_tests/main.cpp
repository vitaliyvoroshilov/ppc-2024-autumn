#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/voroshilov_v_num_of_alphabetic_chars/include/ops_mpi.hpp"

TEST(voroshilov_v_num_of_alphabetic_chars_mpi_perf, test_pipeline_run_mpi) {
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

  boost::mpi::communicator world;
  std::vector<char> global_vec(str.length());
  std::vector<int32_t> global_num(1, initial_num);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (size_t i = 0; i < global_vec.size(); i++) {
      global_vec[i] = str[i];
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_num.data()));
    taskDataPar->outputs_count.emplace_back(global_num.size());
  }

  auto alphabetCharsTaskParallel =
      std::make_shared<voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskParallel>(taskDataPar);
  ASSERT_EQ(alphabetCharsTaskParallel->validation(), true);
  alphabetCharsTaskParallel->pre_processing();
  alphabetCharsTaskParallel->run();
  alphabetCharsTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(alphabetCharsTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(expected_num, global_num[0]);
  }
}

TEST(voroshilov_v_num_of_alphabetic_chars_mpi_perf, test_task_run_mpi) {
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

  boost::mpi::communicator world;
  std::vector<char> global_vec(str.length());
  std::vector<int32_t> global_num(1, initial_num);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (size_t i = 0; i < global_vec.size(); i++) {
      global_vec[i] = str[i];
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_num.data()));
    taskDataPar->outputs_count.emplace_back(global_num.size());
  }

  auto alphabetCharsTaskParallel =
      std::make_shared<voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskParallel>(taskDataPar);
  ASSERT_EQ(alphabetCharsTaskParallel->validation(), true);
  alphabetCharsTaskParallel->pre_processing();
  alphabetCharsTaskParallel->run();
  alphabetCharsTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(alphabetCharsTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(expected_num, global_num[0]);
  }
}
