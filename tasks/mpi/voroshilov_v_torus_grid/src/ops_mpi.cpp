#include "mpi/voroshilov_v_torus_grid/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <thread>
#include <vector>

int voroshilov_v_torus_grid_mpi::select_path_proc(int current_id, int destination_id, int grid) {
  int destination_row_id = destination_id / grid;
  int destination_col_id = destination_id % grid;
  int current_row_id = current_id / grid;
  int current_col_id = current_id % grid;

  int next_row_id = current_row_id;
  int next_col_id = current_col_id;

  if (destination_row_id - current_row_id != 0) {
    // Row is not the same

    // Destination is lower
    if (destination_row_id - current_row_id > 0) {
      if (destination_row_id - current_row_id <= grid / 2) {
        // Direct way
        next_row_id = current_row_id + 1;
        if (next_row_id == grid) {
          next_row_id = 0;
        }
      } else {
        // Reverse way
        next_row_id = current_row_id - 1;
        if (next_row_id == -1) {
          next_row_id = grid - 1;
        }
      }
    }

    // Destination is higher
    if (destination_row_id - current_row_id < 0) {
      if (-1 * (destination_row_id - current_row_id) <= grid / 2) {
        // Direct way
        next_row_id = current_row_id - 1;
        if (next_row_id == -1) {
          next_row_id = grid - 1;
        }
      } else {
        // Reverse way
        next_row_id = current_row_id + 1;
        if (next_row_id == grid) {
          next_row_id = 0;
        }
      }
    }

  } else {
    // Row is the same

    // Destination is on right
    if (destination_col_id - current_col_id > 0) {
      if (destination_col_id - current_col_id <= grid / 2) {
        // Direct way
        next_col_id = current_col_id + 1;
        if (next_col_id == grid) {
          next_col_id = 0;
        }
      } else {
        // Reverse way
        next_col_id = current_col_id - 1;
        if (next_col_id == -1) {
          next_col_id = grid - 1;
        }
      }
    }

    // Destination is on left
    if (destination_col_id - current_col_id < 0) {
      if (-1 * (destination_col_id - current_col_id) <= grid / 2) {
        // Direct way
        next_col_id = current_col_id - 1;
        if (next_col_id == -1) {
          next_col_id = grid - 1;
        }
      } else {
        // Reverse way
        next_col_id = current_col_id + 1;
        if (next_col_id == grid) {
          next_col_id = 0;
        }
      }
    }
  }

  int next_id = next_row_id * grid + next_col_id;

  return next_id;
}

std::pair<int, int> voroshilov_v_torus_grid_mpi::select_terminate_proc(int current_id, int terminate_code, int grid) {
  Commands codes{0, 1, 2, 3, 4};

  int current_row_id = current_id / grid;
  int current_col_id = current_id % grid;

  int next_row_id = current_row_id;
  int next_col_id = current_col_id;

  int next_terminate_code = terminate_code;

  if (terminate_code == codes.direct_terminate) {
    // Go forward in row
    if (current_col_id < grid - 1) {
      // Step right in row
      next_row_id = current_row_id;
      next_col_id = current_col_id + 1;
      next_terminate_code = codes.direct_terminate;
    } else {
      // Step to next row
      next_row_id = current_row_id + 1;
      next_col_id = current_col_id;
      next_terminate_code = codes.reverse_terminate;
    }
  }
  if (terminate_code == codes.reverse_terminate) {
    // Go backward in row
    if (current_col_id > 0) {
      // Step left in row
      next_row_id = current_row_id;
      next_col_id = current_col_id - 1;
      next_terminate_code = codes.reverse_terminate;
    } else {
      // Step to next row
      next_row_id = current_row_id + 1;
      next_col_id = current_col_id;
      next_terminate_code = codes.direct_terminate;
    }
  }

  int next_id = next_row_id * grid + next_col_id;
  std::pair<int, int> next_terminate(next_id, next_terminate_code);
  return next_terminate;
}

bool voroshilov_v_torus_grid_mpi::TorusGridTaskParallel::validation() {
  internal_order_test();
  int world_size = world.size();

  int src = taskData->inputs_count[0];
  int dst = taskData->inputs_count[1];

  if ((src >= world_size) || (dst >= world_size)) {
    return false;
  }

  if (world.rank() == src) {
    if ((taskData->inputs_count[2] <= 0) || (taskData->outputs_count[0] <= 0)) {
      return false;
    };
    // Check if there is n^2 processes to build grid
    for (int i = 0; i <= world_size; i++) {
      if (world_size == i * i) {
        return true;
      }
    }
    return false;
  }
  return true;
}

bool voroshilov_v_torus_grid_mpi::TorusGridTaskParallel::pre_processing() {
  internal_order_test();

  source_proc = taskData->inputs_count[0];
  destination_proc = taskData->inputs_count[1];

  if ((world.rank() == source_proc) || (world.rank() == destination_proc)) {
    message.buffer = std::vector<char>(taskData->inputs_count[2]);
    message.path = std::vector<int>{};
    auto* ptr = reinterpret_cast<char*>(taskData->inputs[0]);
    std::copy(ptr, ptr + taskData->inputs_count[2], message.buffer.begin());
  }
  int proc_count = world.size();
  for (int i = 0; i < proc_count; i++) {
    if (proc_count == i * i) {
      grid_size = i;
    }
  }

  commands = {0, 1, 2, 3, 4};

  tags = {0, 1, 2, 3, 4, 5};

  return true;
}

bool voroshilov_v_torus_grid_mpi::TorusGridTaskParallel::run() {
  internal_order_test();

  if (source_proc == destination_proc) {
    message.path.push_back(source_proc);
    return true;
  }

  if (world.rank() != source_proc) {
    world.recv(boost::mpi::any_source, tags.terminate_command, terminate_command);
    world.recv(boost::mpi::any_source, tags.current_proc, current_proc);
  } else {
    terminate_command = commands.send_from_source;
    current_proc = source_proc;
  }

  if (terminate_command == commands.route_to_dest) {
    // sending data is in progress and we are not source process

    size_t buf_size;
    world.recv(boost::mpi::any_source, tags.buf_size, buf_size);

    message.buffer = std::vector<char>(buf_size);
    world.recv(boost::mpi::any_source, tags.buffer, message.buffer.data(), buf_size);

    size_t path_size;
    world.recv(boost::mpi::any_source, tags.path_size, path_size);

    message.path = std::vector<int>(path_size);
    world.recv(boost::mpi::any_source, tags.path, message.path.data(), path_size);
  }

  if (terminate_command == commands.send_from_source || terminate_command == commands.route_to_dest) {
    // we are source process || we are router (or destination process)

    if (current_proc == destination_proc) {
      message.path.push_back(current_proc);
      terminate_command = commands.move_to_zero;
    } else {
      int next_proc = select_path_proc(current_proc, destination_proc, grid_size);
      message.path.push_back(current_proc);
      world.send(next_proc, tags.terminate_command, commands.route_to_dest);
      world.send(next_proc, tags.current_proc, next_proc);

      world.send(next_proc, tags.buf_size, message.buffer.size());
      world.send(next_proc, tags.buffer, message.buffer.data(), message.buffer.size());
      world.send(next_proc, tags.path_size, message.path.size());
      world.send(next_proc, tags.path, message.path.data(), message.path.size());

      world.recv(boost::mpi::any_source, tags.terminate_command, terminate_command);
      world.recv(boost::mpi::any_source, tags.current_proc, current_proc);
    }
  }

  if (terminate_command == commands.move_to_zero) {
    // sending data is completed and we are moving to process 0
    // in order to start terminating from him but now we are not terminating

    int next_proc = select_path_proc(current_proc, 0, grid_size);
    if (current_proc != next_proc) {
      if (next_proc == 0) {
        world.send(next_proc, tags.terminate_command, commands.direct_terminate);
        world.send(next_proc, tags.current_proc, next_proc);
      } else {
        world.send(next_proc, tags.terminate_command, commands.move_to_zero);
        world.send(next_proc, tags.current_proc, next_proc);
      }
      world.recv(boost::mpi::any_source, tags.terminate_command, terminate_command);
      world.recv(boost::mpi::any_source, tags.current_proc, current_proc);
    } else {
      terminate_command = commands.direct_terminate;
    }
  }

  if (terminate_command == commands.direct_terminate || terminate_command == commands.reverse_terminate) {
    // continue terminating in direct way (right and down) || in reverse way (left and down)

    if (world.rank() == world.size() - 1 && grid_size % 2 == 1) {
      // It is last process to terminate if grid_size is odd number
      return true;
    }
    if (world.rank() == world.size() - grid_size && grid_size % 2 == 0) {
      // It is last process to terminate if grid_size is even number
      return true;
    }
    std::pair<int, int> next_terminate = select_terminate_proc(current_proc, terminate_command, grid_size);
    int next_proc = next_terminate.first;
    int next_terminate_command = next_terminate.second;
    world.send(next_proc, tags.terminate_command, next_terminate_command);
    world.send(next_proc, tags.current_proc, next_proc);
    return true;
  }
  return true;
}

bool voroshilov_v_torus_grid_mpi::TorusGridTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == destination_proc) {
    auto* ptr1 = reinterpret_cast<char*>(taskData->outputs[0]);
    std::copy(message.buffer.begin(), message.buffer.end(), ptr1);

    auto* ptr2 = reinterpret_cast<int*>(taskData->outputs[1]);
    std::copy(message.path.begin(), message.path.end(), ptr2);
  }

  // without barrier() it works incorrectly on large number of processes
  world.barrier();

  return true;
}
