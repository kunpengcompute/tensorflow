# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MatMul coverage for the matrix shapes observed in ``test.log``.

The log was produced from several production models.  It contains operator
input shapes, but not the MatMul transpose attributes, so the one attention
case that requires a transposed right hand side records that attribute
explicitly below.
"""

import numpy as np

from absl.testing import parameterized
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.kernel_tests.benchmark.matmul_cases import MODEL_SHAPES


class MatMulModelShapesTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(*MODEL_SHAPES)
  @test_util.run_without_tensor_float_32("Tests production MatMul shapes")
  def testMatchesNumpy(self, unused_name, shape_a, shape_b, transpose_a,
                       transpose_b):
    rng = np.random.RandomState(20260826)
    a = rng.normal(0.0, 0.1, size=shape_a).astype(np.float32)
    b = rng.normal(0.0, 0.1, size=shape_b).astype(np.float32)

    effective_a = np.swapaxes(a, -1, -2) if transpose_a else a
    effective_b = np.swapaxes(b, -1, -2) if transpose_b else b
    expected = np.matmul(effective_a, effective_b)

    output = self.evaluate(math_ops.matmul(
        a, b, transpose_a=transpose_a, transpose_b=transpose_b))
    self.assertAllClose(expected, output, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
  test.main()
