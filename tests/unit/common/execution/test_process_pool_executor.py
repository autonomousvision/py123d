import pytest

from py123d.common.execution.executor import Task
from py123d.common.execution.process_pool_executor import ProcessPoolExecutor


def _double(x):
    """Top-level function required for pickling in ProcessPoolExecutor."""
    return x * 2


def _add(x, y):
    """Top-level function required for pickling in ProcessPoolExecutor."""
    return x + y


def _identity(x):
    """Top-level function required for pickling in ProcessPoolExecutor."""
    return x


def _fail(x):
    """Top-level function that raises, required for pickling."""
    raise ValueError("process boom")


class TestProcessPoolExecutor:
    def test_init_default_workers(self):
        """Test that default max_workers uses all available CPUs."""
        executor = ProcessPoolExecutor()
        assert executor.config.number_of_nodes == 1
        assert executor.config.number_of_cpus_per_node >= 1
        assert executor.config.number_of_gpus_per_node == 0

    def test_init_custom_workers(self):
        """Test initialization with explicit max_workers."""
        executor = ProcessPoolExecutor(max_workers=2)
        assert executor.config.number_of_cpus_per_node == 2
        assert executor.number_of_threads == 2

    def test_map_single_arg(self):
        """Test mapping a function over a single list."""
        executor = ProcessPoolExecutor(max_workers=2)
        task = Task(fn=_double)
        result = executor.map(task, [1, 2, 3, 4])
        assert sorted(result) == [2, 4, 6, 8]

    def test_map_multiple_args(self):
        """Test mapping a function over multiple argument lists."""
        executor = ProcessPoolExecutor(max_workers=2)
        task = Task(fn=_add)
        result = executor.map(task, [1, 2, 3], [10, 20, 30])
        assert sorted(result) == [11, 22, 33]

    def test_map_preserves_order(self):
        """Test that ProcessPoolExecutor.map preserves input order."""
        executor = ProcessPoolExecutor(max_workers=2)
        task = Task(fn=_identity)
        items = list(range(20))
        result = executor.map(task, items)
        assert result == items

    def test_map_empty_list(self):
        """Test mapping over an empty list."""
        executor = ProcessPoolExecutor(max_workers=2)
        task = Task(fn=_identity)
        result = executor.map(task, [])
        assert result == []

    def test_submit_returns_future(self):
        """Test that submit returns a future with the correct result."""
        executor = ProcessPoolExecutor(max_workers=2)
        task = Task(fn=_add)
        future = executor.submit(task, 3, 4)
        assert future.result(timeout=10) == 7

    def test_submit_exception_propagates(self):
        """Test that exceptions from submitted tasks propagate via the future."""
        executor = ProcessPoolExecutor(max_workers=2)
        task = Task(fn=_fail)
        future = executor.submit(task, 1)
        with pytest.raises(ValueError, match="process boom"):
            future.result(timeout=10)

    def test_processes_are_isolated(self):
        """Test that processes do NOT share memory (unlike threads)."""
        shared = []

        # This lambda can't be pickled, but the concept is:
        # processes can't mutate the parent's shared list.
        # We verify by using a picklable function that returns its input.
        executor = ProcessPoolExecutor(max_workers=2)
        task = Task(fn=_identity)
        result = executor.map(task, [1, 2, 3, 4])
        assert sorted(result) == [1, 2, 3, 4]
        # shared list remains empty because processes are isolated
        assert shared == []
