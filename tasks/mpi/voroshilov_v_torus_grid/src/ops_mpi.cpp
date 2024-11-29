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
  int current_row_id = current_id / grid;
  int current_col_id = current_id % grid;

  int next_row_id = current_row_id;
  int next_col_id = current_col_id;
  
  int next_terminate_code = terminate_code;

  if (terminate_code == 10) {
    // Go forward in row
    if (current_col_id < grid - 1) {
      // Step right in row
      next_row_id = current_row_id;
      next_col_id = current_col_id + 1;
      next_terminate_code = 10;  
    } else {
      // Step to next row
      next_row_id = current_row_id + 1;
      next_col_id = current_col_id;
      next_terminate_code = -10;
    }
  }
  if (terminate_code == -10) {
    // Go backward in row
    if (current_col_id > 0) {
      // Step left in row
      next_row_id = current_row_id;
      next_col_id = current_col_id - 1;
      next_terminate_code = -10;  
    } else {
      // Step to next row
      next_row_id = current_row_id + 1;
      next_col_id = current_col_id;
      next_terminate_code = 10;
    }
  }

  int next_id = next_row_id * grid + next_col_id;
  std::pair<int, int> next_terminate(next_id, next_terminate_code);
  return next_terminate;
}

bool voroshilov_v_torus_grid_mpi::TorusGridTaskParallel::validation() {
  internal_order_test();
  size_t world_size = world.size();

  if ((taskData->inputs_count[0] >= world_size) || (taskData->inputs_count[1] >= world_size)) {
    return false;
  }

  int src = taskData->inputs_count[0];

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

  source_proc = static_cast<int>(taskData->inputs_count[0]);
  destination_proc = static_cast<int>(taskData->inputs_count[1]);

  ///////////////////std::cout << std::endl << world.rank() << " get: " << source_proc << "->" << destination_proc << std::endl;

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
  return true;
}

bool voroshilov_v_torus_grid_mpi::TorusGridTaskParallel::run() {
  internal_order_test();

  if (source_proc == destination_proc) {
    message.path.push_back(source_proc);
    return true;
  }

  if (world.rank() != source_proc) {
    world.recv(boost::mpi::any_source, 1000, terminate_command);
    world.recv(boost::mpi::any_source, 0, current_proc);
  } else {
    terminate_command = 0;
    current_proc = source_proc;
  }

  if (terminate_command == 1) {
    // command 1: sending data is in progress and we are not source process

    size_t buf_size;
    world.recv(boost::mpi::any_source, 1, buf_size);

    message.buffer = std::vector<char>(buf_size);
    world.recv(boost::mpi::any_source, 2, message.buffer.data(), buf_size);

    size_t path_size;
    world.recv(boost::mpi::any_source, 3, path_size);

    message.path = std::vector<int>(path_size);
    world.recv(boost::mpi::any_source, 4, message.path.data(), path_size);

  }

  if (terminate_command == 0 || terminate_command == 1) {
    // command 0: we are source process, command 1: we are router (or destination process)

    if (current_proc == destination_proc) {
      message.path.push_back(current_proc);
      terminate_command = -100;
    } else {
      int next_proc = select_path_proc(current_proc, destination_proc, grid_size);
      message.path.push_back(current_proc);
      world.send(next_proc, 1000, 1);
      world.send(next_proc, 0, next_proc);

      world.send(next_proc, 1, message.buffer.size());
      world.send(next_proc, 2, message.buffer.data(), message.buffer.size());
      world.send(next_proc, 3, message.path.size());
      world.send(next_proc, 4, message.path.data(), message.path.size());

      world.recv(boost::mpi::any_source, 1000, terminate_command);
      world.recv(boost::mpi::any_source, 0, current_proc);
    }
  }

  if (terminate_command == -100) {
    // command -100: sending data is completed and we are moving to process 0 in order to start terminating from him
    //               but now we are not terminating

    int next_proc = select_path_proc(current_proc, 0, grid_size);
    if (current_proc != next_proc) {
      if (next_proc == 0) {
        world.send(next_proc, 1000, 10);
        world.send(next_proc, 0, next_proc);
      } else {
        world.send(next_proc, 1000, -100);
        world.send(next_proc, 0, next_proc);
      }
      world.recv(boost::mpi::any_source, 1000, terminate_command);
      world.recv(boost::mpi::any_source, 0, current_proc);
    } else {
      terminate_command = 10;
    }
  }

  if (terminate_command == 10 || terminate_command == -10) {
    // command 10: we continue terminating in direct way (right and down), command -10: in reverse way (left and down)

    if (world.rank() == world.size() - 1 && grid_size % 2 == 1 ) {
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
    world.send(next_proc, 1000, next_terminate_command);
    world.send(next_proc, 0, next_proc);
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

  return true;
}
