#include "mpi/voroshilov_v_bivariate_optimization_by_area/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <thread>
#include <vector>

// Sequentional:

bool voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskSequential::validation() {
  internal_order_test();

  // criterium-function length <= 0:
  if (taskData->inputs_count[0] <= 0) {
    return false;
  }
  // constraints count is not equal as it is:
  size_t g_count = *reinterpret_cast<size_t*>(taskData->inputs[1]);
  if (g_count != (taskData->inputs).size() - 4) {
    return false;
  }
  // incorrect number of search areas:
  if (taskData->inputs_count[2 + g_count] != 4) {
    return false;
  }
  // search areas:
  auto* d_ptr = reinterpret_cast<double*>(taskData->inputs[2 + g_count]);
  double x_min = *d_ptr++;
  double x_max = *d_ptr++;
  double y_min = *d_ptr++;
  double y_max = *d_ptr;
  if (x_min > x_max) {
    return false;
  }
  if (y_min > y_max) {
    return false;
  }
  // incorrect number of steps count:
  if (taskData->inputs_count[2 + g_count + 1] != 2) {
    return false;
  }
  // steps counts:
  auto* s_ptr = reinterpret_cast<int*>(taskData->inputs[2 + g_count + 1]);
  int x_steps = *s_ptr++;
  int y_steps = *s_ptr;
  // steps_count x:
  if (x_steps <= 0) {
    return false;
  }
  // steps_count y:
  if (y_steps <= 0) {
    return false;
  }

  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskSequential::pre_processing() {
  internal_order_test();

  // criterium-function:
  std::vector<char> q_vec(taskData->inputs_count[0]);
  char* q_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  std::copy(q_ptr, q_ptr + taskData->inputs_count[0], q_vec.begin());
  q = Polynomial(q_vec);

  size_t g_count = *reinterpret_cast<int*>(taskData->inputs[1]);

  // constraints-functions:
  for (size_t i = 2; i < 2 + g_count; i++) {
    char* g_ptr = reinterpret_cast<char*>(taskData->inputs[i]);
    std::vector<char> current_g_vec(taskData->inputs_count[i]);
    std::copy(g_ptr, g_ptr + taskData->inputs_count[i], current_g_vec.begin());
    Polynomial current_g_pol(current_g_vec);
    g.push_back(current_g_pol);
  }

  // search area:
  auto* d_ptr = reinterpret_cast<double*>(taskData->inputs[2 + g_count]);
  double x_min = *d_ptr++;
  double x_max = *d_ptr++;
  double y_min = *d_ptr++;
  double y_max = *d_ptr++;

  // steps counts:
  int* s_ptr = reinterpret_cast<int*>(taskData->inputs[2 + g_count + 1]);
  int x_steps = *s_ptr++;
  int y_steps = *s_ptr;

  x_area = Search_area(x_min, x_max, x_steps);
  y_area = Search_area(y_min, y_max, y_steps);

  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskSequential::run() {
  internal_order_test();

  // Preparing vector of points:

  double x_step = (x_area.max_value - x_area.min_value) / x_area.steps_count;
  double y_step = (y_area.max_value - y_area.min_value) / y_area.steps_count;

  std::vector<Point> points;
  Point current_point(x_area.min_value, y_area.min_value);

  while (current_point.y <= y_area.max_value) {
    while (current_point.x <= x_area.max_value) {
      points.push_back(current_point);
      current_point.x += x_step;
    }
    current_point.x = x_area.min_value;
    current_point.y += y_step;
  }

  // Finding minimum in this vector of points:

  // Find first point satisfied constraints:
  size_t index = 0;
  bool flag_in_area = false;
  while ((index < points.size()) && (!flag_in_area)) {
    // Check if constraints is satisfied:
    flag_in_area = true;
    for (size_t j = 0; j < g.size(); j++) {
      if (g[j].calculate(points[index]) > 0) {
        flag_in_area = false;
        break;
      }
    }
    index++;
  }
  size_t first_satisfied = index;  // it is first candidate for optimum
  optimum_point = points[first_satisfied];
  optimum_value = q.calculate(points[first_satisfied]);

  // Start search from this point:
  for (size_t i = first_satisfied + 1; i < points.size(); i++) {
    double current_value = q.calculate(points[i]);
    // Check if constraints is satisfied:
    flag_in_area = true;
    for (size_t j = 0; j < g.size(); j++) {
      if (g[j].calculate(points[i]) > 0) {
        flag_in_area = false;
        break;
      }
    }
    // Check if current < optimum
    if (flag_in_area) {
      if (current_value < optimum_value) {
        optimum_value = current_value;
        optimum_point.x = points[i].x;
        optimum_point.y = points[i].y;
      }
    }
  }

  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = optimum_value;

  return true;
}

// Parallel:

bool voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    // criterium-function length <= 0:
    if (taskData->inputs_count[0] <= 0) {
      return false;
    }
    // constraints count is not equal as it is:
    size_t g_count = *reinterpret_cast<size_t*>(taskData->inputs[1]);
    if (g_count != (taskData->inputs).size() - 4) {
      return false;
    } 
    /*
    // incorrect number of search areas:
    if (taskData->inputs_count[2 + g_count] != 4) {
      return false;
    }
    // search areas:
    auto* d_ptr = reinterpret_cast<double*>(taskData->inputs[2 + g_count]);
    double x_min = *d_ptr++;
    double x_max = *d_ptr++;
    double y_min = *d_ptr++;
    double y_max = *d_ptr;
    if (x_min > x_max) {
      return false;
    }
    if (y_min > y_max) {
      return false;
    }
    // incorrect number of steps count:
    if (taskData->inputs_count[2 + g_count + 1] != 2) {
      return false;
    }
    // steps counts:
    auto* s_ptr = reinterpret_cast<int*>(taskData->inputs[2 + g_count + 1]);
    int x_steps = *s_ptr++;
    int y_steps = *s_ptr;
    // steps_count x:
    if (x_steps <= 0) {
      return false;
    }
    // steps_count y:
    if (y_steps <= 0) {
      return false;
    }*/
  }
  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskParallel::pre_processing() {
  internal_order_test();

  // criterium-function:
  std::vector<char> q_vec(taskData->inputs_count[0]);
  char* q_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  std::copy(q_ptr, q_ptr + taskData->inputs_count[0], q_vec.begin());
  q = Polynomial(q_vec);

  size_t g_count = *reinterpret_cast<int*>(taskData->inputs[1]);

  // constraints-functions:
  for (size_t i = 2; i < 2 + g_count; i++) {
    char* g_ptr = reinterpret_cast<char*>(taskData->inputs[i]);
    std::vector<char> current_g_vec(taskData->inputs_count[i]);
    std::copy(g_ptr, g_ptr + taskData->inputs_count[i], current_g_vec.begin());
    Polynomial current_g_pol(current_g_vec);
    g.push_back(current_g_pol);
  }

  if (world.rank() == 0) {
    // search area:
    auto* d_ptr = reinterpret_cast<double*>(taskData->inputs[2 + g_count]);
    double x_min = *d_ptr++;
    double x_max = *d_ptr++;
    double y_min = *d_ptr++;
    double y_max = *d_ptr++;

    // steps counts:
    int* s_ptr = reinterpret_cast<int*>(taskData->inputs[2 + g_count + 1]);
    int x_steps = *s_ptr++;
    int y_steps = *s_ptr;

    x_area = Search_area(x_min, x_max, x_steps);
    y_area = Search_area(y_min, y_max, y_steps);
  }

  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskParallel::run() {
  internal_order_test();

  std::vector<Point> points;
  std::vector<double> x_coords;
  std::vector<double> y_coords;

  if (world.rank() == 0) {
    // Preparing vector of points:

    double x_step = (x_area.max_value - x_area.min_value) / x_area.steps_count;
    double y_step = (y_area.max_value - y_area.min_value) / y_area.steps_count;

    Point current_point(x_area.min_value, y_area.min_value);

    while (current_point.y <= y_area.max_value) {
      while (current_point.x <= x_area.max_value) {
        points.push_back(current_point);
        x_coords.push_back(current_point.x);
        y_coords.push_back(current_point.y);
        current_point.x += x_step;
      }
      current_point.x = x_area.min_value;
      current_point.y += y_step;
    }
  }

  // Vector distribution:
  int part;
  int remainder;
  if (world.rank() == 0) {
    part = points.size() / world.size();
    remainder = points.size() % world.size();
  }
  boost::mpi::broadcast(world, part, 0);
  boost::mpi::broadcast(world, remainder, 0);

  std::vector<int> parts(world.size(), part);
  std::vector<int> offsets(world.size());

  for (int i = 0; i < world.size(); i++) {
    if (remainder > 0) {
      parts[i]++;
      remainder--;
    }
    if (i == 0) {
      offsets[i] = 0;
    } else {
      offsets[i] = offsets[i - 1] + parts[i - 1];
    }
  }

  std::vector<double> local_x_coords(parts[world.rank()]);
  std::vector<double> local_y_coords(parts[world.rank()]);
  boost::mpi::scatterv(world, x_coords.data(), parts, offsets, local_x_coords.data(), parts[world.rank()], 0);
  boost::mpi::scatterv(world, y_coords.data(), parts, offsets, local_y_coords.data(), parts[world.rank()], 0);

  std::vector<Point> local_points;
  for (size_t i = 0; i < local_x_coords.size(); i++) {
    Point current_point(local_x_coords[i], local_y_coords[i]);
    local_points.push_back(current_point);
  }

  // Finding minimum in local vector of points:

  // Find first point satisfied constraints:
  size_t index = 0;
  bool flag_in_area = false;
  while ((index < local_points.size()) && (!flag_in_area)) {
    // Check if constraints is satisfied:
    flag_in_area = true;
    for (size_t j = 0; j < g.size(); j++) {
      if (g[j].calculate(local_points[index]) > 0) {
        flag_in_area = false;
        break;
      }
    }
    index++;
  }
  Point local_optimum_point;
  double local_optimum_value;
  size_t first_satisfied = index;  // it is first candidate for optimum
  if (first_satisfied == local_points.size()) {
    local_optimum_point = Point(DBL_MAX, DBL_MAX);
    local_optimum_value = DBL_MAX;
  } else {
    local_optimum_point = local_points[first_satisfied];
    local_optimum_value = q.calculate(local_points[first_satisfied]);
  }

  // Start search from this point:
  for (size_t i = first_satisfied + 1; i < local_points.size(); i++) {
    double current_value = q.calculate(local_points[i]);
    // Check if constraints is satisfied:
    flag_in_area = true;
    for (size_t j = 0; j < g.size(); j++) {
      if (g[j].calculate(local_points[i]) > 0) {
        flag_in_area = false;
        break;
      }
    }
    // Check if current < optimum
    if (flag_in_area) {
      if (current_value < local_optimum_value) {
        local_optimum_value = current_value;
        local_optimum_point.x = local_points[i].x;
        local_optimum_point.y = local_points[i].y;
      }
    }
  }

  boost::mpi::reduce(world, local_optimum_value, global_optimum_value, boost::mpi::minimum<double>(), 0);

  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = global_optimum_value;
  }
  return true;
}
