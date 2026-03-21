from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from py123d.api.map.arrow.arrow_id_utils import IntIDMapping, ToIntMapping, parse_int_id


class TestParseIntId:
    """Tests for parse_int_id covering type handling and edge cases."""

    def test_int_passthrough(self):
        result = parse_int_id(42)
        assert result == 42

    def test_negative_int(self):
        result = parse_int_id(-5)
        assert result == -5

    def test_zero(self):
        result = parse_int_id(0)
        assert result == 0

    def test_large_int(self):
        result = parse_int_id(999_999_999_999)
        assert result == 999_999_999_999

    def test_float_truncates(self):
        result = parse_int_id(3.7)
        assert result == 3

    def test_float_negative(self):
        result = parse_int_id(-2.9)
        assert result == -2

    def test_numeric_string(self):
        result = parse_int_id("123")
        assert result == 123

    def test_negative_numeric_string(self):
        result = parse_int_id("-10")
        assert result == -10

    def test_non_numeric_string_returns_none(self):
        result = parse_int_id("abc")
        assert result is None

    def test_empty_string_returns_none(self):
        result = parse_int_id("")
        assert result is None

    def test_whitespace_string_returns_none(self):
        result = parse_int_id("  ")
        assert result is None

    def test_float_string_returns_none(self):
        """parse_int_id uses int() on strings, so '3.14' raises ValueError and returns None."""
        result = parse_int_id("3.14")
        assert result is None

    def test_unsupported_type_raises_type_error(self):
        with pytest.raises(TypeError, match="ID must be int or string"):
            parse_int_id([1, 2, 3])  # type: ignore[arg-type]

    def test_unsupported_type_none_raises_type_error(self):
        with pytest.raises(TypeError, match="ID must be int or string"):
            parse_int_id(None)  # type: ignore[arg-type]

    def test_bool_treated_as_int(self):
        """bool is a subclass of int in Python, so isinstance(True, int) is True."""
        result = parse_int_id(True)
        assert result == 1


class TestIntIDMapping:
    """Tests for IntIDMapping covering construction, mapping, and edge cases."""

    # --- Construction: from_series ---

    def test_from_series_basic(self):
        series = pd.Series(["a", "b", "c"])
        mapping = IntIDMapping.from_series(series)

        assert len(mapping.str_to_int) == 3
        assert set(mapping.str_to_int.keys()) == {"a", "b", "c"}
        assert set(mapping.str_to_int.values()) == {0, 1, 2}

    def test_from_series_drops_nan(self):
        series = pd.Series(["a", None, "b", float("nan")])
        mapping = IntIDMapping.from_series(series)

        assert "nan" not in mapping.str_to_int
        assert "None" not in mapping.str_to_int
        assert len(mapping.str_to_int) == 2

    def test_from_series_deduplicates(self):
        series = pd.Series(["x", "x", "y", "y", "y"])
        mapping = IntIDMapping.from_series(series)

        assert len(mapping.str_to_int) == 2

    def test_from_series_empty(self):
        series = pd.Series([], dtype=object)
        mapping = IntIDMapping.from_series(series)

        assert mapping.str_to_int == {}
        assert mapping.int_to_str == {}

    def test_from_series_all_nan(self):
        series = pd.Series([None, float("nan"), None])
        mapping = IntIDMapping.from_series(series)

        assert mapping.str_to_int == {}

    def test_from_series_numeric_values_become_strings(self):
        series = pd.Series([1, 2, 3])
        mapping = IntIDMapping.from_series(series)

        assert "1" in mapping.str_to_int
        assert "2" in mapping.str_to_int

    # --- Construction: from_list ---

    def test_from_list_basic(self):
        mapping = IntIDMapping.from_list(["a", "b", "c"])

        assert len(mapping.str_to_int) == 3
        assert set(mapping.str_to_int.keys()) == {"a", "b", "c"}

    def test_from_list_with_int_values(self):
        mapping = IntIDMapping.from_list([10, 20, 30])

        assert "10" in mapping.str_to_int
        assert "20" in mapping.str_to_int

    def test_from_list_does_not_drop_nan_string(self):
        """from_list converts all values via str() first, so str(float('nan')) becomes 'nan' which is NOT NaN."""
        mapping = IntIDMapping.from_list([float("nan"), "a"])

        # str(float('nan')) == 'nan', which is a valid string, not pd.NaN
        assert "nan" in mapping.str_to_int

    def test_from_list_empty(self):
        mapping = IntIDMapping.from_list([])

        assert mapping.str_to_int == {}

    # --- Inverse mapping ---

    def test_int_to_str_inverse(self):
        mapping = IntIDMapping({"alpha": 0, "beta": 1})

        assert mapping.int_to_str == {0: "alpha", 1: "beta"}

    # --- map() ---

    def test_map_known_key(self):
        mapping = IntIDMapping({"foo": 0, "bar": 1})

        result = mapping.map("foo")
        assert result == 0

    def test_map_none_returns_none(self):
        mapping = IntIDMapping({"a": 0})

        result = mapping.map(None)
        assert result is None

    def test_map_nan_returns_none(self):
        mapping = IntIDMapping({"a": 0})

        result = mapping.map(float("nan"))
        assert result is None

    def test_map_numpy_nan_returns_none(self):
        mapping = IntIDMapping({"a": 0})

        result = mapping.map(np.nan)
        assert result is None

    def test_map_unknown_key_returns_none(self):
        mapping = IntIDMapping({"a": 0})

        result = mapping.map("unknown")
        assert result is None

    def test_map_float_key_converts_to_int_string(self):
        """map(1.0) should look up '1', not '1.0'."""
        mapping = IntIDMapping({"1": 0, "2": 1})

        result = mapping.map(1.0)
        assert result == 0

    def test_map_float_key_no_match_when_stored_as_float_string(self):
        """If the key was stored as '1.0', map(1.0) converts to '1' and misses."""
        mapping = IntIDMapping({"1.0": 0})

        result = mapping.map(1.0)
        assert result is None

    def test_map_int_key_as_string(self):
        mapping = IntIDMapping({"42": 0})

        result = mapping.map(42)
        assert result == 0

    # --- map_list() ---

    def test_map_list_none_returns_empty(self):
        mapping = IntIDMapping({"a": 0})

        result = mapping.map_list(None)
        assert result == []

    def test_map_list_filters_unknown_silently(self):
        mapping = IntIDMapping({"a": 0, "b": 1})

        result = mapping.map_list(["a", "unknown", "b"])
        assert result == [0, 1]

    def test_map_list_all_unknown(self):
        mapping = IntIDMapping({"a": 0})

        result = mapping.map_list(["x", "y", "z"])
        assert result == []

    def test_map_list_empty_input(self):
        mapping = IntIDMapping({"a": 0})

        result = mapping.map_list([])
        assert result == []


class TestToIntMapping:
    """Tests for ToIntMapping covering construction, collision handling, and edge cases."""

    # --- Construction: from_list ---

    def test_from_list_all_integers(self):
        mapping = ToIntMapping.from_list([0, 1, 2])

        assert mapping.str_to_int == {"0": 0, "1": 1, "2": 2}

    def test_from_list_all_strings(self):
        mapping = ToIntMapping.from_list(["abc", "def"])

        assert mapping.map("abc") is not None
        assert mapping.map("def") is not None
        # String IDs should get sequential ints starting from 0
        assert set(mapping.str_to_int.values()) == {0, 1}

    def test_from_list_mixed_int_and_string(self):
        mapping = ToIntMapping.from_list([5, "abc", 10, "def"])

        # Integer IDs keep their value
        assert mapping.str_to_int["5"] == 5
        assert mapping.str_to_int["10"] == 10
        # String IDs get next available ints
        abc_val = mapping.str_to_int["abc"]
        def_val = mapping.str_to_int["def"]
        # They should not collide with 5 or 10
        assert abc_val not in {5, 10}
        assert def_val not in {5, 10}
        assert abc_val != def_val

    def test_from_list_empty(self):
        mapping = ToIntMapping.from_list([])

        assert mapping.str_to_int == {}
        assert mapping.int_to_id == {}

    def test_stress_id_collision_int_and_string_same_value(self):
        """STRESS: from_list([1, '1']) -- both parse to int 1. The dict overwrites silently."""
        mapping = ToIntMapping.from_list([1, "1"])

        # Both map to str key "1" with value 1; second overwrites first (same value anyway)
        assert mapping.str_to_int["1"] == 1
        assert len(mapping.str_to_int) == 1

    def test_stress_negative_int_ids(self):
        """STRESS: negative int IDs should be preserved as their int value."""
        mapping = ToIntMapping.from_list([-5, "abc"])

        assert mapping.str_to_int["-5"] == -5
        # "abc" gets next_available_int which starts at 0 (max(-5+1, 0) = 0)
        # But next_available_int = max(0, -5+1) = 0 after first pass
        assert mapping.str_to_int["abc"] == 0

    def test_stress_negative_int_next_available_stays_at_zero(self):
        """STRESS: negative IDs don't advance next_available_int past 0."""
        mapping = ToIntMapping.from_list([-3, -1, "x", "y"])

        assert mapping.str_to_int["-3"] == -3
        assert mapping.str_to_int["-1"] == -1
        # next_available_int = max(0, -3+1, -1+1) = 0
        assert mapping.str_to_int["x"] == 0
        assert mapping.str_to_int["y"] == 1

    def test_stress_very_large_int_id(self):
        """STRESS: large int ID pushes next_available_int very high."""
        mapping = ToIntMapping.from_list([999_999_999, "abc"])

        assert mapping.str_to_int["999999999"] == 999_999_999
        # "abc" gets 1_000_000_000
        assert mapping.str_to_int["abc"] == 1_000_000_000

    def test_stress_duplicate_string_ids(self):
        """STRESS: duplicate non-numeric string IDs in the list. Second pass processes both."""
        mapping = ToIntMapping.from_list(["abc", "abc", "def"])

        # "abc" appears twice in ids_to_remap, but dict assignment overwrites
        # First "abc" gets int 0, second "abc" overwrites with int 1
        assert "abc" in mapping.str_to_int
        assert "def" in mapping.str_to_int
        # All assigned values should be unique (last write wins for "abc")
        values = list(mapping.str_to_int.values())
        assert len(values) == len(set(values)), "Mapped integer values must be unique"

    def test_stress_gap_filling(self):
        """Verify string IDs fill gaps left by integer IDs."""
        mapping = ToIntMapping.from_list([0, 2, "x", "y"])

        assert mapping.str_to_int["0"] == 0
        assert mapping.str_to_int["2"] == 2
        # next_available_int after first pass = max(0+1, 2+1) = 3
        # "x" gets 3, "y" gets 4 (no gap filling because next_available only goes forward)
        assert mapping.str_to_int["x"] == 3
        assert mapping.str_to_int["y"] == 4

    def test_stress_numeric_string_treated_as_integer(self):
        """A numeric string like '7' is parsed as int 7 in the first pass."""
        mapping = ToIntMapping.from_list(["7", "abc"])

        assert mapping.str_to_int["7"] == 7
        assert mapping.str_to_int["abc"] == 8

    def test_stress_float_string_treated_as_non_numeric(self):
        """'3.14' cannot be parsed by int(), so it goes to the second pass as a string ID."""
        mapping = ToIntMapping.from_list(["3.14", "abc"])

        # Both are non-numeric strings, get sequential ints starting from 0
        assert mapping.str_to_int["3.14"] == 0
        assert mapping.str_to_int["abc"] == 1

    # --- Inverse mapping ---

    def test_int_to_id_inverse(self):
        mapping = ToIntMapping({"alpha": 10, "beta": 20})

        assert mapping.int_to_id == {10: "alpha", 20: "beta"}

    # --- map() ---

    def test_map_known_string_key(self):
        mapping = ToIntMapping.from_list([5, "abc"])

        result = mapping.map("abc")
        assert result is not None

    def test_map_none_returns_none(self):
        mapping = ToIntMapping.from_list([1])

        result = mapping.map(None)
        assert result is None

    def test_map_unknown_returns_none(self):
        mapping = ToIntMapping.from_list([1, "abc"])

        result = mapping.map("zzz")
        assert result is None

    def test_map_float_key_converts_to_int_string(self):
        """map(5.0) should look up '5', not '5.0'."""
        mapping = ToIntMapping.from_list([5, "abc"])

        result = mapping.map(5.0)
        assert result == 5

    def test_map_int_key_as_string_lookup(self):
        mapping = ToIntMapping.from_list([42])

        result = mapping.map(42)
        assert result == 42

    # --- map_list() ---

    def test_map_list_none_returns_empty(self):
        mapping = ToIntMapping.from_list([1])

        result = mapping.map_list(None)
        assert result == []

    def test_map_list_filters_unknown(self):
        mapping = ToIntMapping.from_list([1, 2])

        result = mapping.map_list(["1", "999", "2"])
        assert result == [1, 2]

    def test_map_list_empty_input(self):
        mapping = ToIntMapping.from_list([1])

        result = mapping.map_list([])
        assert result == []

    def test_map_list_preserves_order(self):
        mapping = ToIntMapping.from_list([10, 20, 30])

        result = mapping.map_list(["30", "10", "20"])
        assert result == [30, 10, 20]

    # --- Stress: uniqueness invariant ---

    def test_stress_all_mapped_values_unique(self):
        """For any input list, all mapped integer values in str_to_int must be unique."""
        inputs = [0, 1, 5, 100, "abc", "def", "ghi", "0", "5"]
        mapping = ToIntMapping.from_list(inputs)

        values = list(mapping.str_to_int.values())
        assert len(values) == len(set(values)), f"Duplicate int values found: {values}"

    def test_stress_many_string_ids_no_collision(self):
        """Large number of non-numeric string IDs should all get unique ints."""
        ids = [f"id_{i}" for i in range(500)]
        mapping = ToIntMapping.from_list(ids)

        values = list(mapping.str_to_int.values())
        assert len(values) == 500
        assert len(set(values)) == 500

    def test_stress_interleaved_ints_and_strings(self):
        """Integers that occupy 0,1,2,... force string IDs to start after them."""
        mapping = ToIntMapping.from_list([0, 1, 2, "a", "b"])

        assert mapping.str_to_int["0"] == 0
        assert mapping.str_to_int["1"] == 1
        assert mapping.str_to_int["2"] == 2
        assert mapping.str_to_int["a"] == 3
        assert mapping.str_to_int["b"] == 4
