#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Try to implement attention: first build V matrix
    // We need to use the first i+1 values
    Matrix* v_result = matrix_memory_allocator.Allocate("v_concat_" + std::to_string(i));

    if (i == 0) {
      // Just use V[0]
      v_result = values[0];
    } else {
      // Build V by concatenating V[0] to V[i]
      Matrix* temp_v = matrix_memory_allocator.Allocate("temp_v_" + std::to_string(i));
      if (values[0]->GetPosition() != Position::kInGpuHbm) {
        gpu_sim.MoveMatrixToGpuHbm(values[0]);
      }
      gpu_sim.Copy(values[0], temp_v, Position::kInGpuHbm);

      for (size_t j = 1; j <= i; ++j) {
        Matrix* new_concat = matrix_memory_allocator.Allocate("new_v_" + std::to_string(i) + "_" + std::to_string(j));
        if (values[j]->GetPosition() != Position::kInGpuHbm) {
          gpu_sim.MoveMatrixToGpuHbm(values[j]);
        }
        gpu_sim.Concat(temp_v, values[j], new_concat, 0, Position::kInGpuHbm);
        gpu_sim.ReleaseMatrix(temp_v);
        temp_v = new_concat;
      }

      v_result = temp_v;
    }

    // For now, return the concatenated V matrix
    rater.CommitAnswer(*v_result);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu