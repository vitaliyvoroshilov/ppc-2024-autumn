#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace voroshilov_v_torus_grid_mpi {

enum Command { send_from_source = 0, route_to_dest, move_to_zero, direct_terminate, reverse_terminate };

int select_path_proc(int current_id, int destination_id, int grid);
std::pair<int, Command> select_terminate_proc(int current_id, Command terminate_code, int grid);

class TorusGridTaskParallel : public ppc::core::Task {
 public:
  explicit TorusGridTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  struct Message {
    std::vector<char> buffer;
    std::vector<int> path;
  } message;

  int grid_size;

  int source_proc;
  int destination_proc;
  int current_proc;

  Command terminate_command;

  struct Tags {
    static const int terminate_command = 0;
    static const int current_proc = 1;
    static const int buf_size = 2;
    static const int buffer = 3;
    static const int path_size = 4;
    static const int path = 5;
  };

  boost::mpi::communicator world;
};

}  // namespace voroshilov_v_torus_grid_mpi