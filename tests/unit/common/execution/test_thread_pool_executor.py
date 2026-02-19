import pytest

from py123d.common.execution.executor import Task
from py123d.common.execution.thread_pool_executor import ThreadPoolExecutor


class TestThreadPoolExecutor:
    def test_init_default_workers(self):
        """Test that default max_workers uses all available CPUs."""
        executor = ThreadPoolExecutor()
        assert executor.config.number_of_nodes == 1
        assert executor.config.number_of_cpus_per_node >= 1
        assert executor.config.number_of_gpus_per_node == 0

    def test_init_custom_workers(self):
        """Test initialization with explicit max_workers."""
        executor = ThreadPoolExecutor(max_workers=2)
        assert executor.config.number_of_cpus_per_node == 2
        assert executor.number_of_threads == 2

    def test_map_single_arg(self):
        """Test mapping a function over a single list."""
        executor = ThreadPoolExecutor(max_workers=2)
        task = Task(fn=lambda x: x * 2)
        result = executor.map(task, [1, 2, 3, 4])
        assert sorted(result) == [2, 4, 6, 8]

    def test_map_multiple_args(self):
        """Test mapping a function over multiple argument lists."""
        executor = ThreadPoolExecutor(max_workers=2)
        task = Task(fn=lambda x, y: x + y)
        result = executor.map(task, [1, 2, 3], [10, 20, 30])
        assert sorted(result) == [11, 22, 33]

    def test_map_preserves_order(self):
        """Test that ThreadPoolExecutor.map preserves input order."""
        executor = ThreadPoolExecutor(max_workers=2)
        task = Task(fn=lambda x: x)
        items = list(range(20))
        result = executor.map(task, items)
        assert result == items

    def test_map_empty_list(self):
        """Test mapping over an empty list."""
        executor = ThreadPoolExecutor(max_workers=2)
        task = Task(fn=lambda x: x)
        result = executor.map(task, [])
        assert result == []

    def test_submit_returns_future(self):
        """Test that submit returns a future with the correct result."""
        executor = ThreadPoolExecutor(max_workers=2)
        task = Task(fn=lambda x, y: x + y)
        future = executor.submit(task, 3, 4)
        assert future.result(timeout=5) == 7

    def test_submit_exception_propagates(self):
        """Test that exceptions from submitted tasks propagate via the future."""
        executor = ThreadPoolExecutor(max_workers=2)

        def fail():
            raise ValueError("thread boom")

        task = Task(fn=fail)
        future = executor.submit(task)
        with pytest.raises(ValueError, match="thread boom"):
            future.result(timeout=5)

    def test_map_with_shared_state(self):
        """Test that threads share memory (can append to same list)."""
        shared = []

        def append_item(x):
            shared.append(x)
            return x

        executor = ThreadPoolExecutor(max_workers=2)
        task = Task(fn=append_item)
        executor.map(task, [1, 2, 3, 4])
        assert sorted(shared) == [1, 2, 3, 4]
