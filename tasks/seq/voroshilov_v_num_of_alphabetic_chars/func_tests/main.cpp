#include <gtest/gtest.h>

#include <vector>
#include <string>

#include "seq/voroshilov_v_num_of_alphabetic_chars/include/ops_seq.hpp"

TEST(voroshilov_v_num_of_alphabetic_chars_seq_func, test_without_alphabetic_chars_seq) {
  std::string str = "123456789-+*/=<>";
  int initial_num = 0;
  int expected_num = 0;

  // Create data
  std::vector<char> in(str.length());
  for (size_t i = 0; i < in.size(); i++)
    in[i] = str[i];
  std::vector<int> out(1, initial_num);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential alphabetCharsTaskSequential(taskDataSeq);
  ASSERT_EQ(alphabetCharsTaskSequential.validation(), true);
  alphabetCharsTaskSequential.pre_processing();
  alphabetCharsTaskSequential.run();
  alphabetCharsTaskSequential.post_processing();
  ASSERT_EQ(expected_num, out[0]);
}

TEST(voroshilov_v_num_of_alphabetic_chars_seq_func, test_with_lowercase_alphabetic_chars_seq) {
  std::string str = "123456789-+*/=<>aaabbcxyyzzz";
  int initial_num = 0, expected_num = 12;

  // Create data
  std::vector<char> in(str.length());
  for (size_t i = 0; i < in.size(); i++)
    in[i] = str[i];
  std::vector<int> out(1, initial_num);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential alphabetCharsTaskSequential(taskDataSeq);
  ASSERT_EQ(alphabetCharsTaskSequential.validation(), true);
  alphabetCharsTaskSequential.pre_processing();
  alphabetCharsTaskSequential.run();
  alphabetCharsTaskSequential.post_processing();
  ASSERT_EQ(expected_num, out[0]);
}

TEST(voroshilov_v_num_of_alphabetic_chars_seq_func, test_with_uppercase_alphabetic_chars_seq) {
  std::string str = "123456789-+*/=<>AAABBCXYYZZZ";
  int initial_num = 0, expected_num = 12;

  // Create data
  std::vector<char> in(str.length());
  for (size_t i = 0; i < in.size(); i++)
    in[i] = str[i];
  std::vector<int> out(1, initial_num);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential alphabetCharsTaskSequential(taskDataSeq);
  ASSERT_EQ(alphabetCharsTaskSequential.validation(), true);
  alphabetCharsTaskSequential.pre_processing();
  alphabetCharsTaskSequential.run();
  alphabetCharsTaskSequential.post_processing();
  ASSERT_EQ(expected_num, out[0]);
}

TEST(voroshilov_v_num_of_alphabetic_chars_seq_func, test_with_anycase_alphabetic_chars_seq) {
  std::string str = "123456789-+*/=<>aaabbcxyyzzzAAABBCXYYZZZ";
  int initial_num = 0, expected_num = 24;

  // Create data
  std::vector<char> in(str.length());
  for (size_t i = 0; i < in.size(); i++)
    in[i] = str[i];
  std::vector<int> out(1, initial_num);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential alphabetCharsTaskSequential(taskDataSeq);
  ASSERT_EQ(alphabetCharsTaskSequential.validation(), true);
  alphabetCharsTaskSequential.pre_processing();
  alphabetCharsTaskSequential.run();
  alphabetCharsTaskSequential.post_processing();
  ASSERT_EQ(expected_num, out[0]);
}
