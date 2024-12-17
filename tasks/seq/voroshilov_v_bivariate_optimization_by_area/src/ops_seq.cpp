#include "seq/voroshilov_v_bivariate_optimization_by_area/include/ops_seq.hpp"

#include <thread>

bool voroshilov_v_bivariate_optimization_by_area_seq::OptimizationTaskSequential::validation() {
  internal_order_test();

  // criterium-function length < 0:
  if (taskData->inputs_count[0] < 0) {
    return false;
  }
  // min_x > max_x:
  if (taskData->inputs[1][0] > taskData->inputs[1][1]) {
    return false;
  }
  // steps_count x:
  if (taskData->inputs[1][2] < 0) {
    return false;
  }
  // min_y > max_y:
  if (taskData->inputs[1][3] > taskData->inputs[1][4]) {
    return false;
  }
  // steps_count y:
  if (taskData->inputs[1][5] < 0) {
    return false;
  }
  // constraints functions count is incorrect:
  if (inputs_count[1] != inputs.size() - 2) {
    return false;
  }

  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_seq::OptimizationTaskSequential::pre_processing() {
  internal_order_test();

  // criterium-function:
  q = std::vector<char>(taskData->inputs_count[0]);
  char* q_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  std::copy(q_ptr, q_ptr + taskData->inputs_count[0], q.begin());

  // search area, steps_counts:
  int* d_ptr = reinterpret_cast<int*>(taskData->inputs[1]);
  x_area.min_value = *d_ptr++;
  x_area.max_value = *d_ptr++;
  x_area.steps_count = *d_ptr++;
  y_area.min_value = *d_ptr++;
  y_area.max_value = *d_ptr++;
  y_area.steps_count = *d_ptr;

  // constraints-functions:
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
  
  std::vector<std::vector<Point>> points(y_area.steps_count);
  int x_step = (x_area.max_value - x_area.min_value) / x_area.steps_count;
  int y_step = (y_area.max_value - y_area.min_value) / y_area.steps_count;

  

  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_seq::OptimizationTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
