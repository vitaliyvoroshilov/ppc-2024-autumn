#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/voroshilov_v_torus_grid/include/ops_mpi.hpp"

struct Func_test_tags {
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

TEST(voroshilov_v_torus_grid_mpi_func, test_validation_empty_input_mpi) {
  int data_size = 0;

  boost::mpi::communicator world;

  int src_proc = 0;
  int dst_proc = world.size() - 1;

  std::vector<char> input_data(0);
  std::vector<char> output_data(data_size);
  std::vector<int> output_path;

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

  voroshilov_v_torus_grid_mpi::TorusGridTaskParallel torusGridTaskParallel(taskDataPar);
  ASSERT_EQ(torusGridTaskParallel.validation(), false);
}

TEST(voroshilov_v_torus_grid_mpi_func, test_validation_src_process_not_exists_mpi) {
  int data_size = 100000;

  boost::mpi::communicator world;

  int src_proc = world.size();
  int dst_proc = 0;

  std::vector<char> input_data = generate_data(data_size);
  std::vector<char> output_data(data_size);
  std::vector<int> output_path;

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

  voroshilov_v_torus_grid_mpi::TorusGridTaskParallel torusGridTaskParallel(taskDataPar);
  ASSERT_EQ(torusGridTaskParallel.validation(), false);
}

TEST(voroshilov_v_torus_grid_mpi_func, test_validation_dst_process_not_exists_mpi) {
  int data_size = 100000;

  boost::mpi::communicator world;

  int src_proc = 0;
  int dst_proc = world.size();

  std::vector<char> input_data = generate_data(data_size);
  std::vector<char> output_data(data_size);
  std::vector<int> output_path;

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

  voroshilov_v_torus_grid_mpi::TorusGridTaskParallel torusGridTaskParallel(taskDataPar);
  ASSERT_EQ(torusGridTaskParallel.validation(), false);
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_first_to_first_mpi) {
  int data_size = 100000;

  boost::mpi::communicator world;

  int src_proc = 0;
  int dst_proc = 0;

  std::vector<char> input_data(data_size);
  std::vector<char> output_data(data_size);
  std::vector<int> expected_path = calculate_expected_path(src_proc, dst_proc, world.size());
  std::vector<int> output_path(expected_path.size());

  if (world.rank() == src_proc) {
    input_data = generate_data(data_size);
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.send(dst_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }
  if (world.rank() == dst_proc) {
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.recv(src_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }

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

  voroshilov_v_torus_grid_mpi::TorusGridTaskParallel torusGridTaskParallel(taskDataPar);
  ASSERT_EQ(torusGridTaskParallel.validation(), true);
  torusGridTaskParallel.pre_processing();
  torusGridTaskParallel.run();
  torusGridTaskParallel.post_processing();

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
      world.send(0, Func_test_tags::send_flag_data, flag_data);
      world.send(0, Func_test_tags::send_flag_path, flag_path);
    }
  }

  if (world.rank() == 0) {
    bool flg_data = true;
    bool flg_path = true;
    if ((world.size() > 1) && (dst_proc != 0)) {
      world.recv(dst_proc, Func_test_tags::send_flag_data, flg_data);
      world.recv(dst_proc, Func_test_tags::send_flag_path, flg_path);
    }
    ASSERT_EQ(flg_data, true);
    ASSERT_EQ(flg_path, true);
  }
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_first_to_middle_mpi) {
  int data_size = 100000;

  boost::mpi::communicator world;

  int src_proc = 0;
  int dst_proc = world.size() / 2;

  std::vector<char> input_data(data_size);
  std::vector<char> output_data(data_size);
  std::vector<int> expected_path = calculate_expected_path(src_proc, dst_proc, world.size());
  std::vector<int> output_path(expected_path.size());

  if (world.rank() == src_proc) {
    input_data = generate_data(data_size);
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.send(dst_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }
  if (world.rank() == dst_proc) {
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.recv(src_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }

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

  voroshilov_v_torus_grid_mpi::TorusGridTaskParallel torusGridTaskParallel(taskDataPar);
  ASSERT_EQ(torusGridTaskParallel.validation(), true);
  torusGridTaskParallel.pre_processing();
  torusGridTaskParallel.run();
  torusGridTaskParallel.post_processing();

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
      world.send(0, Func_test_tags::send_flag_data, flag_data);
      world.send(0, Func_test_tags::send_flag_path, flag_path);
    }
  }

  if (world.rank() == 0) {
    bool flg_data = true;
    bool flg_path = true;
    if ((world.size() > 1) && (dst_proc != 0)) {
      world.recv(dst_proc, Func_test_tags::send_flag_data, flg_data);
      world.recv(dst_proc, Func_test_tags::send_flag_path, flg_path);
    }
    ASSERT_EQ(flg_data, true);
    ASSERT_EQ(flg_path, true);
  }
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_first_to_last_mpi) {
  int data_size = 100000;

  boost::mpi::communicator world;

  int src_proc = 0;
  int dst_proc = world.size() - 1;

  std::vector<char> input_data(data_size);
  std::vector<char> output_data(data_size);
  std::vector<int> expected_path = calculate_expected_path(src_proc, dst_proc, world.size());
  std::vector<int> output_path(expected_path.size());

  if (world.rank() == src_proc) {
    input_data = generate_data(data_size);
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.send(dst_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }
  if (world.rank() == dst_proc) {
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.recv(src_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }

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

  voroshilov_v_torus_grid_mpi::TorusGridTaskParallel torusGridTaskParallel(taskDataPar);
  ASSERT_EQ(torusGridTaskParallel.validation(), true);
  torusGridTaskParallel.pre_processing();
  torusGridTaskParallel.run();
  torusGridTaskParallel.post_processing();

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
      world.send(0, Func_test_tags::send_flag_data, flag_data);
      world.send(0, Func_test_tags::send_flag_path, flag_path);
    }
  }

  if (world.rank() == 0) {
    bool flg_data = true;
    bool flg_path = true;
    if ((world.size() > 1) && (dst_proc != 0)) {
      world.recv(dst_proc, Func_test_tags::send_flag_data, flg_data);
      world.recv(dst_proc, Func_test_tags::send_flag_path, flg_path);
    }
    ASSERT_EQ(flg_data, true);
    ASSERT_EQ(flg_path, true);
  }
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_middle_to_first_mpi) {
  int data_size = 100000;

  boost::mpi::communicator world;

  int src_proc = world.size() / 2;
  int dst_proc = 0;

  std::vector<char> input_data(data_size);
  std::vector<char> output_data(data_size);
  std::vector<int> expected_path = calculate_expected_path(src_proc, dst_proc, world.size());
  std::vector<int> output_path(expected_path.size());

  if (world.rank() == src_proc) {
    input_data = generate_data(data_size);
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.send(dst_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }
  if (world.rank() == dst_proc) {
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.recv(src_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }

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

  voroshilov_v_torus_grid_mpi::TorusGridTaskParallel torusGridTaskParallel(taskDataPar);
  ASSERT_EQ(torusGridTaskParallel.validation(), true);
  torusGridTaskParallel.pre_processing();
  torusGridTaskParallel.run();
  torusGridTaskParallel.post_processing();

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
      world.send(0, Func_test_tags::send_flag_data, flag_data);
      world.send(0, Func_test_tags::send_flag_path, flag_path);
    }
  }

  if (world.rank() == 0) {
    bool flg_data = true;
    bool flg_path = true;
    if ((world.size() > 1) && (dst_proc != 0)) {
      world.recv(dst_proc, Func_test_tags::send_flag_data, flg_data);
      world.recv(dst_proc, Func_test_tags::send_flag_path, flg_path);
    }
    ASSERT_EQ(flg_data, true);
    ASSERT_EQ(flg_path, true);
  }
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_middle_to_middle_mpi) {
  int data_size = 10000;

  boost::mpi::communicator world;

  int src_proc = world.size() / 2;
  int dst_proc = world.size() / 2;

  std::vector<char> input_data(data_size);
  std::vector<char> output_data(data_size);
  std::vector<int> expected_path = calculate_expected_path(src_proc, dst_proc, world.size());
  std::vector<int> output_path(expected_path.size());

  if (world.rank() == src_proc) {
    input_data = generate_data(data_size);
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.send(dst_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }
  if (world.rank() == dst_proc) {
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.recv(src_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }

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

  voroshilov_v_torus_grid_mpi::TorusGridTaskParallel torusGridTaskParallel(taskDataPar);
  ASSERT_EQ(torusGridTaskParallel.validation(), true);
  torusGridTaskParallel.pre_processing();
  torusGridTaskParallel.run();
  torusGridTaskParallel.post_processing();

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
      world.send(0, Func_test_tags::send_flag_data, flag_data);
      world.send(0, Func_test_tags::send_flag_path, flag_path);
    }
  }

  if (world.rank() == 0) {
    bool flg_data = true;
    bool flg_path = true;
    if ((world.size() > 1) && (dst_proc != 0)) {
      world.recv(dst_proc, Func_test_tags::send_flag_data, flg_data);
      world.recv(dst_proc, Func_test_tags::send_flag_path, flg_path);
    }
    ASSERT_EQ(flg_data, true);
    ASSERT_EQ(flg_path, true);
  }
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_middle_to_last_mpi) {
  int data_size = 100000;

  boost::mpi::communicator world;

  int src_proc = world.size() / 2;
  int dst_proc = world.size() - 1;

  std::vector<char> input_data(data_size);
  std::vector<char> output_data(data_size);
  std::vector<int> expected_path = calculate_expected_path(src_proc, dst_proc, world.size());
  std::vector<int> output_path(expected_path.size());

  if (world.rank() == src_proc) {
    input_data = generate_data(data_size);
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.send(dst_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }
  if (world.rank() == dst_proc) {
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.recv(src_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }

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

  voroshilov_v_torus_grid_mpi::TorusGridTaskParallel torusGridTaskParallel(taskDataPar);
  ASSERT_EQ(torusGridTaskParallel.validation(), true);
  torusGridTaskParallel.pre_processing();
  torusGridTaskParallel.run();
  torusGridTaskParallel.post_processing();

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
      world.send(0, Func_test_tags::send_flag_data, flag_data);
      world.send(0, Func_test_tags::send_flag_path, flag_path);
    }
  }

  if (world.rank() == 0) {
    bool flg_data = true;
    bool flg_path = true;
    if ((world.size() > 1) && (dst_proc != 0)) {
      world.recv(dst_proc, Func_test_tags::send_flag_data, flg_data);
      world.recv(dst_proc, Func_test_tags::send_flag_path, flg_path);
    }
    ASSERT_EQ(flg_data, true);
    ASSERT_EQ(flg_path, true);
  }
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_last_to_first_mpi) {
  int data_size = 100000;

  boost::mpi::communicator world;

  int src_proc = world.size() - 1;
  int dst_proc = 0;

  std::vector<char> input_data(data_size);
  std::vector<char> output_data(data_size);
  std::vector<int> expected_path = calculate_expected_path(src_proc, dst_proc, world.size());
  std::vector<int> output_path(expected_path.size());

  if (world.rank() == src_proc) {
    input_data = generate_data(data_size);
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.send(dst_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }
  if (world.rank() == dst_proc) {
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.recv(src_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }

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

  voroshilov_v_torus_grid_mpi::TorusGridTaskParallel torusGridTaskParallel(taskDataPar);
  ASSERT_EQ(torusGridTaskParallel.validation(), true);
  torusGridTaskParallel.pre_processing();
  torusGridTaskParallel.run();
  torusGridTaskParallel.post_processing();

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
      world.send(0, Func_test_tags::send_flag_data, flag_data);
      world.send(0, Func_test_tags::send_flag_path, flag_path);
    }
  }

  if (world.rank() == 0) {
    bool flg_data = true;
    bool flg_path = true;
    if ((world.size() > 1) && (dst_proc != 0)) {
      world.recv(dst_proc, Func_test_tags::send_flag_data, flg_data);
      world.recv(dst_proc, Func_test_tags::send_flag_path, flg_path);
    }
    ASSERT_EQ(flg_data, true);
    ASSERT_EQ(flg_path, true);
  }
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_last_to_middle_mpi) {
  int data_size = 100000;

  boost::mpi::communicator world;

  int src_proc = world.size() - 1;
  int dst_proc = world.size() / 2;

  std::vector<char> input_data(data_size);
  std::vector<char> output_data(data_size);
  std::vector<int> expected_path = calculate_expected_path(src_proc, dst_proc, world.size());
  std::vector<int> output_path(expected_path.size());

  if (world.rank() == src_proc) {
    input_data = generate_data(data_size);
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.send(dst_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }
  if (world.rank() == dst_proc) {
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.recv(src_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }

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

  voroshilov_v_torus_grid_mpi::TorusGridTaskParallel torusGridTaskParallel(taskDataPar);
  ASSERT_EQ(torusGridTaskParallel.validation(), true);
  torusGridTaskParallel.pre_processing();
  torusGridTaskParallel.run();
  torusGridTaskParallel.post_processing();

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
      world.send(0, Func_test_tags::send_flag_data, flag_data);
      world.send(0, Func_test_tags::send_flag_path, flag_path);
    }
  }

  if (world.rank() == 0) {
    bool flg_data = true;
    bool flg_path = true;
    if ((world.size() > 1) && (dst_proc != 0)) {
      world.recv(dst_proc, Func_test_tags::send_flag_data, flg_data);
      world.recv(dst_proc, Func_test_tags::send_flag_path, flg_path);
    }
    ASSERT_EQ(flg_data, true);
    ASSERT_EQ(flg_path, true);
  }
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_last_to_last_mpi) {
  int data_size = 100000;

  boost::mpi::communicator world;

  int src_proc = world.size() - 1;
  int dst_proc = world.size() - 1;

  std::vector<char> input_data(data_size);
  std::vector<char> output_data(data_size);
  std::vector<int> expected_path = calculate_expected_path(src_proc, dst_proc, world.size());
  std::vector<int> output_path(expected_path.size());

  if (world.rank() == src_proc) {
    input_data = generate_data(data_size);
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.send(dst_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }
  if (world.rank() == dst_proc) {
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.recv(src_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }

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

  voroshilov_v_torus_grid_mpi::TorusGridTaskParallel torusGridTaskParallel(taskDataPar);
  ASSERT_EQ(torusGridTaskParallel.validation(), true);
  torusGridTaskParallel.pre_processing();
  torusGridTaskParallel.run();
  torusGridTaskParallel.post_processing();

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
      world.send(0, Func_test_tags::send_flag_data, flag_data);
      world.send(0, Func_test_tags::send_flag_path, flag_path);
    }
  }

  if (world.rank() == 0) {
    bool flg_data = true;
    bool flg_path = true;
    if ((world.size() > 1) && (dst_proc != 0)) {
      world.recv(dst_proc, Func_test_tags::send_flag_data, flg_data);
      world.recv(dst_proc, Func_test_tags::send_flag_path, flg_path);
    }
    ASSERT_EQ(flg_data, true);
    ASSERT_EQ(flg_path, true);
  }
}

TEST(voroshilov_v_torus_grid_mpi_func, test_run_random_to_random_mpi) {
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
      world.send(dst_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
    }
  }
  if (world.rank() == dst_proc) {
    if ((world.size() > 1) && (src_proc != dst_proc)) {
      world.recv(src_proc, Func_test_tags::send_generated_data, input_data.data(), input_data.size());
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

  voroshilov_v_torus_grid_mpi::TorusGridTaskParallel torusGridTaskParallel(taskDataPar);
  ASSERT_EQ(torusGridTaskParallel.validation(), true);
  torusGridTaskParallel.pre_processing();
  torusGridTaskParallel.run();
  torusGridTaskParallel.post_processing();

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
      world.send(0, Func_test_tags::send_flag_data, flag_data);
      world.send(0, Func_test_tags::send_flag_path, flag_path);
    }
  }

  if (world.rank() == 0) {
    bool flg_data = true;
    bool flg_path = true;
    if ((world.size() > 1) && (dst_proc != 0)) {
      world.recv(dst_proc, Func_test_tags::send_flag_data, flg_data);
      world.recv(dst_proc, Func_test_tags::send_flag_path, flg_path);
    }
    ASSERT_EQ(flg_data, true);
    ASSERT_EQ(flg_path, true);
  }
}
