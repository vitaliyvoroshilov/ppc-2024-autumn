#include "seq/voroshilov_v_bivariate_optimization_by_area/include/ops_seq.hpp"

#include <thread>

int voroshilov_v_bivariate_optimization_by_area_seq::calculate_function(std::string function, Point point) {
  return 0;
}

bool voroshilov_v_bivariate_optimization_by_area_seq::OptimizationTaskSequential::validation() {
  internal_order_test();

  for (int i = 0; i < 2; i++) {
    if (taskData->inputs_count[i] < 0) {
      return false;
    } 
  }
  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_seq::OptimizationTaskSequential::pre_processing() {
  internal_order_test();

  q = std::vector<char>(taskData->inputs_count[0]);
  char* q_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  std::copy(q_ptr, q_ptr + taskData->inputs_count[0], q.begin());

  int* d_ptr = reinterpret_cast<int*>(taskData->inputs[1]);
  x_area.min_value = *d_ptr++;
  x_area.max_value = *d_ptr++;
  y_area.min_value = *d_ptr++;
  y_area.max_value = *d_ptr;

  g = std::vector<std::vector<char>>(taskData->inputs_count[1]);
  for (int i = 2; i < 2 + (g.size() - 1); i++) {
    char* g_ptr = reinterpret_cast<char*>(taskData->inputs[i]);
    std::vector<char> current_g(taskData->inputs_count[i]);
    std::copy(g_ptr, g_ptr + taskData->inputs_count[i], current_g.begin());
    g.emplace_back(current_g);
  }

  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_seq::OptimizationTaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < input_; i++) {
    res++;
  }
  std::this_thread::sleep_for(20ms);
  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_seq::OptimizationTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
