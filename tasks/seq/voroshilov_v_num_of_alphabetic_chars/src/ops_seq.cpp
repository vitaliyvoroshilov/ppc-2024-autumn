#include "seq/voroshilov_v_num_of_alphabetic_chars/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

std::vector<char> voroshilov_v_num_of_alphabetic_chars_seq::genVecWithFixedAlphabeticsCount(int alphCount,
                                                                                            size_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<char> vector(size);
  int curCount = 0;

  std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&(){}[]*+-/";
  int charset_alphabet_size = 52;

  // Generate with absolutely random alphabetics count:
  for (size_t i = 0; i < vector.size(); i++) {
    int number = gen() % charset.length();
    vector[i] = charset[number];
    if (std::isalpha(vector[i]) != 0) {
      curCount++;
    }
  }

  // Change non-alphabetics to alphabetics to complete missing quantity
  for (size_t i = 0; curCount < alphCount; i++) {
    if (std::isalpha(vector[i]) == 0) {
      int number = gen() % charset_alphabet_size;
      vector[i] = charset[number];
      curCount++;
    }
  }

  // Change alphabetics to non-alphabetics if there is an oversupply
  for (size_t i = 0; curCount > alphCount; i++) {
    if (std::isalpha(vector[i]) != 0) {
      int number = gen() % (charset.length() - charset_alphabet_size) + charset_alphabet_size;
      vector[i] = charset[number];
      curCount--;
    }
  }

  return vector;
}

bool voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential::validation() {
  internal_order_test();
  // Check count elements of input and output
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::vector<char>(taskData->inputs_count[0]);
  char* ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  std::copy(ptr, ptr + taskData->inputs_count[0], input_.begin());
  res_ = 0;
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size(); i++) {
    if (std::isalpha(input_[i]) != 0) {  // Check if it is alphabetic character
      res_++;
    }
  }
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = res_;
  return true;
}
