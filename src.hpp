#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Build attention output by concatenating values
    Matrix* result = matrix_memory_allocator.Allocate("attention_" + std::to_string(i));

    if (i == 0) {
      // For round 0, return V[0] as-is (it's already in HBM)
      result = values[0];
    } else {
      // Build result by copying values[0] and then concatenating others
      Matrix* temp = matrix_memory_allocator.Allocate("temp_v_" + std::to_string(i));
      Matrix* current = matrix_memory_allocator.Allocate("current_v_" + std::to_string(i));

      // Start with V[0]
      gpu_sim.Copy(values[0], temp, Position::kInGpuHbm);

      // Concatenate remaining values
      for (size_t j = 1; j <= i; ++j) {
        Matrix* next = matrix_memory_allocator.Allocate("next_v_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.Concat(temp, values[j], next, 0, Position::kInGpuHbm);

        // Update pointers for next iteration
        gpu_sim.ReleaseMatrix(temp);

        // Make sure the result is properly positioned
        gpu_sim.Copy(next, current, Position::kInGpuHbm);

        gpu_sim.ReleaseMatrix(next);
        temp = current;
        current = matrix_memory_allocator.Allocate("current_v_" + std::to_string(i));
      }

      result = temp;
    }

    // Commit the result
    rater.CommitAnswer(*result);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu