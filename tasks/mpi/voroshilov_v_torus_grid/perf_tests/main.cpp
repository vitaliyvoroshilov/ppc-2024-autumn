#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/voroshilov_v_torus_grid/include/ops_mpi.hpp"

struct Perf_test_tags {
  static const int send_generated_data = 111;
  static const int send_flag_data = 222;
  static const int send_flag_path = 333;
};

int generate_rank(int world_size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  return gen() % world_size;
}

std::vector<char> generate_data(size_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<char> vector(size);

  std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&(){}[]*+-/";

  for (size_t i = 0; i < vector.size(); i++) {
    int number = gen() % charset.length();
    vector[i] = charset[number];
  }

  return vector;
}

std::vector<int> calculate_expected_path(int source_id, int destination_id, int processes_count) {
  int grid = 0;
  for (int i = 0; i <= processes_count; i++) {
    if (i * i == processes_count) {
      grid = i;
    }
  }

  std::vector<int> path;
  int next_id = source_id;
  path.push_back(next_id);
  while (next_id != destination_id) {
    next_id = voroshilov_v_torus_grid_mpi::select_path_proc(next_id, destination_id, grid);
    path.push_back(next_id);
  }

  return path;
}

TEST(voroshilov_v_torus_grid_mpi_perf, test_pipeline_run_mpi) {
  int data_size = 100000;

  boost::mpi::communicator world;
  
  std::vector<char> input_data(data_size);
  std::vector<char> output_data(data_size);
  
  int src_proc = 0;
  int dst_proc = 0;

  if (world.rank() == 0) {
    src_proc = generate_rank(world.size());
    dst_proc = generate_rank(world.size());
  }
  // broadcast() because in other way each process generates its own local src_proc and dst_proc
  boost::mpi::broadcast(world, src_proc, 0);
  boost::mpi::broadcast(world, dst_proc, 0);
  
  if (world.rank() == src_proc) {
    input_data = generate_data(data_size);
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.send(dst_proc, Perf_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }
  if (world.rank() == dst_proc) {
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.recv(src_proc, Perf_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }

  std::vector<int> expected_path = calculate_expected_path(src_proc, dst_proc, world.size());
  std::vector<int> output_path(expected_path.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(src_proc);
  taskDataPar->inputs_count.emplace_back(dst_proc);

  if ((world.rank() == src_proc) || (world.rank() == dst_proc)) {

    taskDataPar->inputs_count.emplace_back(input_data.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    
    taskDataPar->outputs_count.emplace_back(output_data.size());
    
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_path.data()));
  }

  auto torusGridTaskParallel = std::make_shared<voroshilov_v_torus_grid_mpi::TorusGridTaskParallel>(taskDataPar);
  ASSERT_EQ(torusGridTaskParallel->validation(), true);
  torusGridTaskParallel->pre_processing();
  torusGridTaskParallel->run();
  torusGridTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(torusGridTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  // ASSERT_EQ() doesnt work if world.rank() != 0 so decided send to process 0 and check there
  if (world.rank() == dst_proc) {
    bool flag_data = true;
    for (int i = 0; i < output_data.size(); i++) {
      if (output_data[i] != input_data[i]) {
        flag_data = false;
      }
    }
    bool flag_path = true;
    for (int i = 0; i < output_path.size(); i++) {
      if (output_path[i] != expected_path[i]) {
        flag_path = false;
      }
    }
    if ((world.size() > 1) && (dst_proc != 0)) {
      world.send(0, Perf_test_tags::send_flag_data, flag_data);
      world.send(0, Perf_test_tags::send_flag_path, flag_path);
    }
  }

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    bool flg_data = true;
    bool flg_path = true;
    if ((world.size() > 1) && (dst_proc != 0)) {
      world.recv(dst_proc, Perf_test_tags::send_flag_data, flg_data);
      world.recv(dst_proc, Perf_test_tags::send_flag_path, flg_path);
    }
    ASSERT_EQ(flg_data, true);
    ASSERT_EQ(flg_path, true);
  }
}

TEST(voroshilov_v_torus_grid_mpi_perf, test_task_run_mpi) {
  int data_size = 100000;

  boost::mpi::communicator world;
  
  std::vector<char> input_data(data_size);
  std::vector<char> output_data(data_size);
  
  int src_proc = 0;
  int dst_proc = 0;

  if (world.rank() == 0) {
    src_proc = generate_rank(world.size());
    dst_proc = generate_rank(world.size());
  }
  // broadcast() because in other way each process generates its own local src_proc and dst_proc
  boost::mpi::broadcast(world, src_proc, 0);
  boost::mpi::broadcast(world, dst_proc, 0);
  
  if (world.rank() == src_proc) {
    input_data = generate_data(data_size);
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.send(dst_proc, Perf_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }
  if (world.rank() == dst_proc) {
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.recv(src_proc, Perf_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }

  std::vector<int> expected_path = calculate_expected_path(src_proc, dst_proc, world.size());
  std::vector<int> output_path(expected_path.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(src_proc);
  taskDataPar->inputs_count.emplace_back(dst_proc);

  if ((world.rank() == src_proc) || (world.rank() == dst_proc)) {

    taskDataPar->inputs_count.emplace_back(input_data.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    
    taskDataPar->outputs_count.emplace_back(output_data.size());
    
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_path.data()));
  }

  auto torusGridTaskParallel = std::make_shared<voroshilov_v_torus_grid_mpi::TorusGridTaskParallel>(taskDataPar);
  ASSERT_EQ(torusGridTaskParallel->validation(), true);
  torusGridTaskParallel->pre_processing();
  torusGridTaskParallel->run();
  torusGridTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(torusGridTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  // ASSERT_EQ() doesnt work if world.rank() != 0 so decided send to process 0 and check there
  if (world.rank() == dst_proc) {
    bool flag_data = true;
    for (int i = 0; i < output_data.size(); i++) {
      if (output_data[i] != input_data[i]) {
        flag_data = false;
      }
    }
    bool flag_path = true;
    for (int i = 0; i < output_path.size(); i++) {
      if (output_path[i] != expected_path[i]) {
        flag_path = false;
      }
    }
    if ((world.size() > 1) && (dst_proc != 0)) {
      world.send(0, Perf_test_tags::send_flag_data, flag_data);
      world.send(0, Perf_test_tags::send_flag_path, flag_path);
    }
  }

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    bool flg_data = true;
    bool flg_path = true;
    if ((world.size() > 1) && (dst_proc != 0)) {
      world.recv(dst_proc, Perf_test_tags::send_flag_data, flg_data);
      world.recv(dst_proc, Perf_test_tags::send_flag_path, flg_path);
    }
    ASSERT_EQ(flg_data, true);
    ASSERT_EQ(flg_path, true);
  }
}
