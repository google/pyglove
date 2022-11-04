# Copyright 2022 The PyGlove Authors
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
"""Tests for pyglove.symbolic.flags."""

import unittest
from pyglove.core.symbolic import flags


class GlobalFlagsTest(unittest.TestCase):
  """Tests for global flags."""

  def test_allow_empty_field_description(self):
    # Default set to True.
    self.assertTrue(flags.is_empty_field_description_allowed())

    flags.allow_empty_field_description(False)
    self.assertFalse(flags.is_empty_field_description_allowed())

    flags.allow_empty_field_description(True)
    self.assertTrue(flags.is_empty_field_description_allowed())

  def test_allow_repeated_class_registration(self):
    # Default set to True.
    self.assertTrue(flags.is_repeated_class_registration_allowed())

    flags.allow_repeated_class_registration(False)
    self.assertFalse(flags.is_repeated_class_registration_allowed())

    flags.allow_repeated_class_registration(True)
    self.assertTrue(flags.is_repeated_class_registration_allowed())

  def test_origin_stacktrace_limit(self):
    # Default set to True.
    self.assertEqual(flags.get_origin_stacktrace_limit(), 10)

    flags.set_origin_stacktrace_limit(5)
    self.assertEqual(flags.get_origin_stacktrace_limit(), 5)

  def test_load_handler(self):
    # Default set to None
    self.assertIsNone(flags.get_load_handler())

    mock_load_handler = lambda *args: None
    self.assertIsNone(flags.set_load_handler(mock_load_handler))
    self.assertIs(flags.get_load_handler(), mock_load_handler)
    self.assertIs(flags.set_load_handler(None), mock_load_handler)
    self.assertIsNone(flags.get_load_handler())

  def test_save_handler(self):
    # Default set to None
    self.assertIsNone(flags.get_save_handler())

    mock_save_handler = lambda *args: None
    self.assertIsNone(flags.set_save_handler(mock_save_handler))
    self.assertIs(flags.get_save_handler(), mock_save_handler)
    self.assertIs(flags.set_save_handler(None), mock_save_handler)
    self.assertIsNone(flags.get_save_handler())


class ScopedFlagsTest(unittest.TestCase):
  """Tests for scoped flags (thread-local)."""

  def test_track_origin(self):
    self.assertFalse(flags.is_tracking_origin())
    with flags.track_origin(True):
      self.assertTrue(flags.is_tracking_origin())
      with flags.track_origin(False):
        self.assertFalse(flags.is_tracking_origin())
      self.assertTrue(flags.is_tracking_origin())
    self.assertFalse(flags.is_tracking_origin())

  def test_notify_on_change(self):
    self.assertTrue(flags.is_change_notification_enabled())
    with flags.notify_on_change(False):
      self.assertFalse(flags.is_change_notification_enabled())
      with flags.notify_on_change(True):
        self.assertTrue(flags.is_change_notification_enabled())
      self.assertFalse(flags.is_change_notification_enabled())
    self.assertTrue(flags.is_change_notification_enabled())

  def test_enable_type_check(self):
    self.assertTrue(flags.is_type_check_enabled())
    with flags.enable_type_check(False):
      self.assertFalse(flags.is_type_check_enabled())
      with flags.enable_type_check(True):
        self.assertTrue(flags.is_type_check_enabled())
      self.assertFalse(flags.is_type_check_enabled())
    self.assertTrue(flags.is_type_check_enabled())

  def test_allow_writable_accessors(self):
    self.assertFalse(flags.is_under_accessor_writable_scope())
    with flags.allow_writable_accessors(True):
      self.assertTrue(flags.is_under_accessor_writable_scope())
      with flags.allow_writable_accessors(False):
        self.assertFalse(flags.is_under_accessor_writable_scope())
      self.assertTrue(flags.is_under_accessor_writable_scope())
    self.assertFalse(flags.is_under_accessor_writable_scope())

  def test_as_sealed(self):
    self.assertFalse(flags.is_under_sealed_scope())
    with flags.as_sealed(True):
      self.assertTrue(flags.is_under_sealed_scope())
      with flags.as_sealed(False):
        self.assertFalse(flags.is_under_sealed_scope())
      self.assertTrue(flags.is_under_sealed_scope())
    self.assertFalse(flags.is_under_sealed_scope())

  def test_allow_partial(self):
    self.assertFalse(flags.is_under_partial_scope())
    with flags.allow_partial(True):
      self.assertTrue(flags.is_under_partial_scope())
      with flags.allow_partial(False):
        self.assertFalse(flags.is_under_partial_scope())
      self.assertTrue(flags.is_under_partial_scope())
    self.assertFalse(flags.is_under_partial_scope())


if __name__ == '__main__':
  unittest.main()
