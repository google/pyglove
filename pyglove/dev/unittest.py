# Copyright 2025 The PyGlove Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unittest runner for Colab notebook."""

import sys
from typing import Callable, Optional, Sequence, Type
import unittest


TestCase = unittest.TestCase


def run_tests(
    test_case_cls: Type[TestCase],
    test_names: Optional[Sequence[str]] = None
) -> None:
  """Run tests."""
  suite = unittest.TestLoader().loadTestsFromTestCase(test_case_cls)
  if test_names:
    test_ids = set()
    for name in test_names:
      test_id = '%s.%s.%s' % (
          test_case_cls.__module__, test_case_cls.__name__, name)
      test_ids.add(test_id)

    tests = []
    for case in suite:
      if not hasattr(case, 'id') or case.id() in test_ids:
        tests.append(case)
    suite = unittest.TestSuite()
    suite.addTests(tests)
  unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)


def enable_test(
    should_run: bool = True,
    test_names: Optional[Sequence[str]] = None
) -> Callable[[Type[TestCase]], Type[TestCase]]:
  """Class decorator that automatically runs test."""
  def _decorator(cls):
    if should_run:
      run_tests(cls, test_names)
    return cls
  return _decorator

