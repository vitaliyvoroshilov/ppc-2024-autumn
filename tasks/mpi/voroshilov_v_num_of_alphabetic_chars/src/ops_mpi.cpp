#include "mpi/voroshilov_v_num_of_alphabetic_chars/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskSequential::validation() {
  internal_order_test();
  // Check count elements of input and output
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::vector<char>(taskData->inputs_count[0]);
  char* ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = ptr[i];
  }
  res_ = 0;
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size(); i++) {
    if (std::isalpha(input_[i])) {  // Check if it is alphabetic character
      res_++;
    }
  }
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = res_;
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of input and output
    return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskParallel::pre_processing() {
  internal_order_test();
  size_t part = 0;
  size_t part_last = 0;
  if (world.rank() == 0) {
    part = taskData->inputs_count[0] / world.size();
    part_last = taskData->inputs_count[0] - part * (world.size() - 1);
    world.send(world.size() - 1, 0, &part_last, 1);
  }
  boost::mpi::broadcast(world, part, 0);

  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<char>(taskData->inputs_count[0]);
    char* ptr = reinterpret_cast<char*>(taskData->inputs[0]);
    for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = ptr[i];
    }
    for (int proc = 1; proc < world.size() - 1; proc++) {
      world.send(proc, 0, input_.data() + proc * part, part);
    }
    world.send(world.size() - 1, 0, input_.data() + (world.size() - 1) * part, part_last);
  }
  local_input_ = std::vector<char>(part);
  
  if (world.rank() == 0) {
    local_input_ = std::vector<char>(input_.begin(), input_.begin() + part);
  } else if (world.rank() == world.size() - 1) {
    world.recv(0, 0, &part_last, 1);
    local_input_.resize(part_last);
    world.recv(0, 0, local_input_.data(), part_last);
  } else {
    world.recv(0, 0, local_input_.data(), part);
  }
  // Init value for output
  res_ = 0;
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskParallel::run() {
  internal_order_test();
  int local_res = 0;
  for (size_t i = 0; i < local_input_.size(); i++) {
    if (std::isalpha(local_input_[i])) {  // Check if it is alphabetic character
      local_res++;
    }
  }
  boost::mpi::reduce(world, local_res, res_, std::plus(), 0);
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<int*>(taskData->outputs[0]) = res_;
  }
  return true;
}
