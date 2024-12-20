#include <gtest/gtest.h>

#include "seq/voroshilov_v_bivariate_optimization_by_area/include/ops_seq.hpp"

bool validation_test(std::vector<char> q_vec, std::vector<double> areas_vec, std::vector<int> steps_vec, int g_count,
                     std::vector<std::vector<char>> g_vec) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  // Criterium-function:
  taskDataSeq->inputs_count.emplace_back(q_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(q_vec.data()));
  // Search area boundaries:
  taskDataSeq->inputs_count.emplace_back(areas_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(areas_vec.data()));
  // Steps counts (how many points will be used):
  taskDataSeq->inputs_count.emplace_back(steps_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(steps_vec.data()));
  // Count of constraints-functions:
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&g_count));
  // Constraints-functions:
  taskDataSeq->inputs_count.emplace_back(g_vec.size());
  for (int i = 0; i < g_vec.size(); i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(g_vec[i].data()));
  }
  // Output - optimum point and value:
  std::vector<double> optimum_vec(3);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(optimum_vec.data()));

  voroshilov_v_bivariate_optimization_by_area_seq::OptimizationTaskSequential optimizationTaskSequential(taskDataSeq);
  return optimizationTaskSequential.validation();
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_validation_empty_criterium_function) {
  std::string q_str = "";
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-10.0, 10.0, -10.0, 10.0});
  std::vector<int> steps_vec({1000, 1000});

  std::string g_str1 = "-x^1y^0";
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1";
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  int g_count = g_vec.size();

  ASSERT_FALSE(validation_test(q_vec, areas_vec, steps_vec, g_count, g_vec));
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_validation_incorrect_num_of_areas) {
  std::string q_str = "x^2y^0+x^0y^2";
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-10.0, 10.0, -10.0});
  std::vector<int> steps_vec({1000, 1000});

  std::string g_str1 = "-x^1y^0";
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1";
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  int g_count = g_vec.size();

  ASSERT_FALSE(validation_test(q_vec, areas_vec, steps_vec, g_count, g_vec));
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_validation_incorrect_areas) {
  std::string q_str = "x^2y^0+x^0y^2";
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-10.0, -20.0, -10.0, 10.0});
  std::vector<int> steps_vec({1000, 1000});

  std::string g_str1 = "-x^1y^0";
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1";
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  int g_count = g_vec.size();

  ASSERT_FALSE(validation_test(q_vec, areas_vec, steps_vec, g_count, g_vec));
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_validation_incorrect_num_of_steps_counts) {
  std::string q_str = "x^2y^0+x^0y^2";
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-10.0, 10.0, -10.0, 10.0});
  std::vector<int> steps_vec({1000});

  std::string g_str1 = "-x^1y^0";
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1";
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  int g_count = g_vec.size();

  ASSERT_FALSE(validation_test(q_vec, areas_vec, steps_vec, g_count, g_vec));
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_validation_incorrect_steps_counts) {
  std::string q_str = "x^2y^0+x^0y^2";
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-10.0, 10.0, -10.0, 10.0});
  std::vector<int> steps_vec({0, 1000});

  std::string g_str1 = "-x^1y^0";
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1";
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  int g_count = g_vec.size();

  ASSERT_FALSE(validation_test(q_vec, areas_vec, steps_vec, g_count, g_vec));
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_validation_incorrect_g_count) {
  std::string q_str = "x^2y^0+x^0y^2";
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-10.0, 10.0, -10.0, 10.0});
  std::vector<int> steps_vec({1000, 1000});

  std::string g_str1 = "-x^1y^0";
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1";
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  int g_count = 33;

  ASSERT_FALSE(validation_test(q_vec, areas_vec, steps_vec, g_count, g_vec));
}

std::vector<double> task_run_test(std::vector<char> q_vec, std::vector<double> areas_vec, std::vector<int> steps_vec,
                                  int g_count, std::vector<std::vector<char>> g_vec) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  // Criterium-function:
  taskDataSeq->inputs_count.emplace_back(q_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(q_vec.data()));
  // Search area boundaries:
  taskDataSeq->inputs_count.emplace_back(areas_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(areas_vec.data()));
  // Steps counts (how many points will be used):
  taskDataSeq->inputs_count.emplace_back(steps_vec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(steps_vec.data()));
  // Count of constraints-functions:
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&g_count));
  // Constraints-functions:
  for (int i = 0; i < g_vec.size(); i++) {
    taskDataSeq->inputs_count.emplace_back(g_vec[i].size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(g_vec[i].data()));
  }
  // Output - optimum point and value:
  std::vector<double> optimum_vec(3);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(optimum_vec.data()));

  voroshilov_v_bivariate_optimization_by_area_seq::OptimizationTaskSequential optimizationTaskSequential(taskDataSeq);
  optimizationTaskSequential.validation();
  optimizationTaskSequential.pre_processing();
  optimizationTaskSequential.run();
  optimizationTaskSequential.post_processing();

  return optimum_vec;
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_task_run_zero_func_without_constraints) {
  std::string q_str = "0";
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-10.0, 10.0, -10.0, 10.0});
  std::vector<int> steps_vec({1000, 1000});

  std::vector<std::vector<char>> g_vec;
  int g_count = g_vec.size();

  std::vector<double> optimum_vec = task_run_test(q_vec, areas_vec, steps_vec, g_count, g_vec);

  double optimum_value = optimum_vec[2];

  double eps = 0.1;

  ASSERT_NEAR(optimum_value, 0.0, eps);  // dont check optimum point because it is everywhere
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_task_run_zero_func_with_constraints) {
  std::string q_str = "0";
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-10.0, 10.0, -10.0, 10.0});
  std::vector<int> steps_vec({1000, 1000});

  std::string g_str1 = "-x^1y^0 +1";  // x >= 1
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1 +1";  // y >= 1
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  int g_count = g_vec.size();

  std::vector<double> optimum_vec = task_run_test(q_vec, areas_vec, steps_vec, g_count, g_vec);

  double optimum_x = optimum_vec[0];
  double optimum_y = optimum_vec[1];
  double optimum_value = optimum_vec[2];

  double eps = 0.1;

  ASSERT_TRUE(optimum_x >= 1.0);
  ASSERT_TRUE(optimum_y >= 1.0);
  ASSERT_NEAR(optimum_value, 0.0, eps);
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_task_run_paraboloid_without_constraints) {
  std::string q_str = "x^2y^0 +x^0y^2";  // paraboloid x^2+y^2, increases from point (0;0)
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-5.0, 5.0, -5.0, 5.0});
  std::vector<int> steps_vec({1000, 1000});

  std::vector<std::vector<char>> g_vec;
  int g_count = g_vec.size();

  std::vector<double> optimum_vec = task_run_test(q_vec, areas_vec, steps_vec, g_count, g_vec);

  double optimum_x = optimum_vec[0];
  double optimum_y = optimum_vec[1];
  double optimum_value = optimum_vec[2];

  double eps = 0.1;

  ASSERT_NEAR(optimum_x, 0.0, eps);
  ASSERT_NEAR(optimum_y, 0.0, eps);
  ASSERT_NEAR(optimum_value, 0.0, eps);
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_task_run_paraboloid_with_constraints) {
  std::string q_str = "x^2y^0 +x^0y^2";  // paraboloid x^2+y^2, increases from point (0;0)
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-5.0, 5.0, -5.0, 5.0});
  std::vector<int> steps_vec({1000, 1000});

  std::string g_str1 = "-x^1y^0 +1";  // x >= 1
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1 +1";  // y >= 1
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  int g_count = g_vec.size();

  std::vector<double> optimum_vec = task_run_test(q_vec, areas_vec, steps_vec, g_count, g_vec);

  double optimum_x = optimum_vec[0];
  double optimum_y = optimum_vec[1];
  double optimum_value = optimum_vec[2];

  double eps = 0.1;

  ASSERT_NEAR(optimum_x, 1.0, eps);
  ASSERT_NEAR(optimum_y, 1.0, eps);
  ASSERT_NEAR(optimum_value, 2.0, eps);
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_task_run_paraboloid_minus_number_without_constraints) {
  std::string q_str = "x^2y^0 +x^0y^2 -10";  // paraboloid x^2+y^2 minus 10 (increases from -10)
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-5.0, 5.0, -5.0, 5.0});
  std::vector<int> steps_vec({1000, 1000});

  std::vector<std::vector<char>> g_vec;
  int g_count = g_vec.size();

  std::vector<double> optimum_vec = task_run_test(q_vec, areas_vec, steps_vec, g_count, g_vec);

  double optimum_x = optimum_vec[0];
  double optimum_y = optimum_vec[1];
  double optimum_value = optimum_vec[2];

  double eps = 0.1;

  ASSERT_NEAR(optimum_x, 0.0, eps);
  ASSERT_NEAR(optimum_y, 0.0, eps);
  ASSERT_NEAR(optimum_value, -10.0, eps);
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_task_run_paraboloid_minus_number_with_constraints) {
  std::string q_str = "x^2y^0 +x^0y^2 -10";  // paraboloid x^2+y^2 munus 10 (increases from -10)
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-5.0, 5.0, -5.0, 5.0});
  std::vector<int> steps_vec({1000, 1000});

  std::string g_str1 = "-x^1y^0 +1";  // x >= 1
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1 +1";  // y >= 1
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  int g_count = g_vec.size();

  std::vector<double> optimum_vec = task_run_test(q_vec, areas_vec, steps_vec, g_count, g_vec);

  double optimum_x = optimum_vec[0];
  double optimum_y = optimum_vec[1];
  double optimum_value = optimum_vec[2];

  double eps = 0.1;

  ASSERT_NEAR(optimum_x, 1.0, eps);
  ASSERT_NEAR(optimum_y, 1.0, eps);
  ASSERT_NEAR(optimum_value, -8.0, eps);
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_task_run_shifted_paraboloid_without_constraints) {
  std::string q_str = "x^2y^0 -12x^1y^0 +x^0y^2 -4x^0y^1";  // shifted paraboloid, increases from about -40
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-10.0, 10.0, -10.0, 10.0});
  std::vector<int> steps_vec({1000, 1000});

  std::vector<std::vector<char>> g_vec;
  int g_count = g_vec.size();

  std::vector<double> optimum_vec = task_run_test(q_vec, areas_vec, steps_vec, g_count, g_vec);

  double optimum_x = optimum_vec[0];
  double optimum_y = optimum_vec[1];
  double optimum_value = optimum_vec[2];

  double eps = 0.1;

  ASSERT_TRUE((0.0 <= optimum_x) && (optimum_x <= 10.0));
  ASSERT_TRUE((0.0 <= optimum_y) && (optimum_x <= 10.0));
  ASSERT_NEAR(optimum_value, -40.0, eps);
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_task_run_shifted_paraboloid_with_constraints) {
  std::string q_str = "x^2y^0 -12x^1y^0 +x^0y^2 -4x^0y^1";  // shifted paraboloid, increases from about -40
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-10.0, 10.0, -10.0, 10.0});
  std::vector<int> steps_vec({1000, 1000});

  std::string g_str1 = "x^0y^1 -2x^1y^0 -4";
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "x^0y^1 +x^1y^0 -4";
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::string g_str3 = "0.2x^1y^0 -x^0y^1 +0.4";
  std::vector<char> g_vec3(g_str3.length());
  std::copy(g_str3.begin(), g_str3.end(), g_vec3.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2, g_vec3});
  int g_count = g_vec.size();

  std::vector<double> optimum_vec = task_run_test(q_vec, areas_vec, steps_vec, g_count, g_vec);

  double optimum_x = optimum_vec[0];
  double optimum_y = optimum_vec[1];
  double optimum_value = optimum_vec[2];

  double eps = 0.1;

  ASSERT_TRUE((-2.0 <= optimum_x) && (optimum_x <= 4.0));
  ASSERT_TRUE((0.0 <= optimum_y) && (optimum_x <= 4.0));
  ASSERT_NEAR(optimum_value, -30.0, eps);
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_task_run_large_degrees_without_constraints) {
  std::string q_str = "x^256y^0 +x^0y^888 +x^100y^28";  // "box", increases from value 0
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-2.0, 2.0, -2.0, 2.0});
  std::vector<int> steps_vec({1000, 1000});

  std::vector<std::vector<char>> g_vec;
  int g_count = g_vec.size();

  std::vector<double> optimum_vec = task_run_test(q_vec, areas_vec, steps_vec, g_count, g_vec);

  double optimum_x = optimum_vec[0];
  double optimum_y = optimum_vec[1];
  double optimum_value = optimum_vec[2];

  double eps = 0.1;

  ASSERT_TRUE((-1.0 <= optimum_x) && (optimum_x <= 1.0));
  ASSERT_TRUE((-1.0 <= optimum_y) && (optimum_y <= 1.0));
  ASSERT_NEAR(optimum_value, 0.0, eps);
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_task_run_large_degrees_with_constraints) {
  std::string q_str = "x^256y^0 +x^0y^888 +x^100y^28";  // "box", increases from value 0
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-2.0, 2.0, -2.0, 2.0});
  std::vector<int> steps_vec({1000, 1000});

  std::string g_str1 = "-x^1y^0 +0.5";  // x >= 0.5
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1 +0.7";  // y >= 0.7
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  int g_count = g_vec.size();

  std::vector<double> optimum_vec = task_run_test(q_vec, areas_vec, steps_vec, g_count, g_vec);

  double optimum_x = optimum_vec[0];
  double optimum_y = optimum_vec[1];
  double optimum_value = optimum_vec[2];

  double eps = 0.1;

  ASSERT_TRUE(0.5 <= optimum_x);
  ASSERT_TRUE(0.7 <= optimum_y);
  ASSERT_NEAR(optimum_value, 0.0, eps);
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_task_run_odd_degrees_without_constraints) {
  std::string q_str = "x^33y^0 +x^0y^51";  // "increasing-decreasing box", increases from 0 where x > ~0.75 or y > ~0.75
                                           // decreases from 0 where x < ~-0.75 or y < ~-0.75
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({0.0, 2.0, 0.0, 2.0});  // areas changed to find min!!!
  std::vector<int> steps_vec({1000, 1000});

  std::vector<std::vector<char>> g_vec;
  int g_count = g_vec.size();

  std::vector<double> optimum_vec = task_run_test(q_vec, areas_vec, steps_vec, g_count, g_vec);

  double optimum_x = optimum_vec[0];
  double optimum_y = optimum_vec[1];
  double optimum_value = optimum_vec[2];

  double eps = 0.1;

  ASSERT_TRUE((0.0 <= optimum_x) && (optimum_x <= 1.0));
  ASSERT_TRUE((0.0 <= optimum_y) && (optimum_y <= 1.0));
  ASSERT_NEAR(optimum_value, 0.0, eps);
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_task_run_odd_degrees_with_constraints) {
  std::string q_str = "x^33y^0 +x^0y^51";  // "increasing-decreasing box", increases from 0 where x > ~0.75 or y > ~0.75
                                           // decreases from 0 where x < ~-0.75 or y < ~-0.75
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-2.0, 2.0, -2.0, 2.0});
  std::vector<int> steps_vec({1000, 1000});

  std::string g_str1 = "-x^1y^0 -0.5";  // x >= -0.5
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1 -0.5";  // y >= -0.5
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2});
  int g_count = g_vec.size();

  std::vector<double> optimum_vec = task_run_test(q_vec, areas_vec, steps_vec, g_count, g_vec);

  double optimum_x = optimum_vec[0];
  double optimum_y = optimum_vec[1];
  double optimum_value = optimum_vec[2];

  double eps = 0.1;

  ASSERT_TRUE((-0.5 <= optimum_x) && (optimum_x <= 1.0));
  ASSERT_TRUE((-0.5 <= optimum_y) && (optimum_y <= 1.0));
  ASSERT_NEAR(optimum_value, 0.0, eps);
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_task_run_negative_degree_without_constraints) {
  std::string q_str = "x^-2y^0";  // "3D hyperbole" with positive values
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-15.0, 0.0, -10.0, 10.0});  // areas changed to find min!!!
  std::vector<int> steps_vec({1000, 1000});

  std::vector<std::vector<char>> g_vec;
  int g_count = g_vec.size();

  std::vector<double> optimum_vec = task_run_test(q_vec, areas_vec, steps_vec, g_count, g_vec);

  double optimum_value = optimum_vec[2];

  double eps = 0.1;

  ASSERT_NEAR(optimum_value, 0.0, eps);
}

TEST(voroshilov_v_bivariate_optimization_by_area_seq_func, test_task_run_negative_degree_with_constraints) {
  std::string q_str = "x^-2y^0";  // "3D hyperbole" with positive values
  std::vector<char> q_vec(q_str.length());
  std::copy(q_str.begin(), q_str.end(), q_vec.begin());

  std::vector<double> areas_vec({-15.0, 0.0, -10.0, 10.0});  // areas changed to find min!!!
  std::vector<int> steps_vec({1000, 1000});

  std::string g_str1 = "-x^1y^0 -10";  // x >= -10
  std::vector<char> g_vec1(g_str1.length());
  std::copy(g_str1.begin(), g_str1.end(), g_vec1.begin());
  std::string g_str2 = "-x^0y^1 -0.5";  // y >= -0.5
  std::vector<char> g_vec2(g_str2.length());
  std::copy(g_str2.begin(), g_str2.end(), g_vec2.begin());
  std::string g_str3 = "x^0y^1 -0.5";  // y <= 0.5
  std::vector<char> g_vec3(g_str3.length());
  std::copy(g_str3.begin(), g_str3.end(), g_vec3.begin());
  std::vector<std::vector<char>> g_vec({g_vec1, g_vec2, g_vec3});
  int g_count = g_vec.size();

  std::vector<double> optimum_vec = task_run_test(q_vec, areas_vec, steps_vec, g_count, g_vec);

  double optimum_x = optimum_vec[0];
  double optimum_y = optimum_vec[1];
  double optimum_value = optimum_vec[2];

  double eps = 0.1;

  ASSERT_TRUE(-10.0 <= optimum_x);
  ASSERT_TRUE((-0.5 <= optimum_y) && (optimum_y <= 0.5));
  ASSERT_NEAR(optimum_value, 0.0, eps);
}
