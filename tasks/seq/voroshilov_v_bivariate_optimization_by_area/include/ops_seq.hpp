#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace voroshilov_v_bivariate_optimization_by_area_seq {

struct Monomial {
public:
  int coef;
  int deg_x;
  int deg_y;

  Monomial(int coef_, int deg_x_, int deg_y_) {
    coef = coef_;
    deg_x = deg_x_;
    deg_y = deg_y_;
  }

  Monomial(std::vector<char> monom) {
    size_t i = 0;
    std::string str_coef = "";
    while (monom[i] != 'x') {
      str_coef += monom[i];
      i++;
    }
    i += 2;

    std::string str_degx = "";
    while (monom[i] != 'y') {
      str_degx += monom[i];
      i++;
    }
    i += 2;

    std::string str_degy = "";
    while (i < monom.size()) {
      str_degy += monom[i];
      i++;
    }

    coef = std::stoi(str_coef);
    deg_x = std::stoi(str_degx);
    deg_y = std::stoi(str_degy);
  }
};

struct Polynomial {
public:
  int length;
  std::vector<Monomial> monomials;

  Polynomial(int length_, std::vector<Monomial> monomials_) {
    length = length_;
    monomials = monomials_;
  }

  Polynomial(std::vector<char> polynom) {
    length = 0;
    size_t i = 0;
    while (i < polynom.size()) {
      std::vector<char> monom;
      monom.push_back(polynom[i]);
      i++;
      while ((i < polynom.size()) && (polynom[i] != '+') && (polynom[i] != '-')) {
          monom.push_back(polynom[i]);
          i++;
      }
      Monomial mnm(monom);
      monomials.push_back(mnm);
      length++;
    }
  }
};

struct Search_area {
public:
  int min_value;
  int max_value;

  Search_area(int min_value_ = 0, int max_value_ = 0) {
    min_value = min_value_;
    max_value = max_value_;
  }
};

struct Point {
public:
  int x;
  int y;

  Point(int x_ = 0, int y_ = 0) {
    x = x_;
    y = y_;
  }
};

int calculate_function(std::string function, Point point);

class OptimizationTaskSequential : public ppc::core::Task {
 public:
  explicit OptimizationTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> q; // criterion function
  std::vector<std::vector<char>> g; // constraints functions

  Search_area x_area;
  Search_area y_area;

  Point optimum_point;
  int optimum_value;
};

}  // namespace voroshilov_v_bivariate_optimization_by_area_seq