import pytest

from py123d.common.execution.executor import Task
from py123d.common.execution.sequential_executor import SequentialExecutor


class TestSequentialExecutor:
    def test_init_resources(self):
        """Test that SequentialExecutor is configured with 1 CPU, 1 node, 0 GPUs."""
        executor = SequentialExecutor()
        assert executor.config.number_of_nodes == 1
        assert executor.config.number_of_cpus_per_node == 1
        assert executor.config.number_of_gpus_per_node == 0
        assert executor.number_of_threads == 1

    def test_map_single_arg(self):
        """Test mapping a function over a single list of arguments."""
        executor = SequentialExecutor()
        task = Task(fn=lambda x: x * 2)
        result = executor.map(task, [1, 2, 3])
        assert result == [2, 4, 6]

    def test_map_multiple_args(self):
        """Test mapping a function over multiple argument lists."""
        executor = SequentialExecutor()
        task = Task(fn=lambda x, y: x + y)
        result = executor.map(task, [1, 2, 3], [10, 20, 30])
        assert result == [11, 22, 33]

    def test_map_preserves_order(self):
        """Test that sequential execution preserves input order."""
        executor = SequentialExecutor()
        task = Task(fn=lambda x: x)
        items = list(range(100))
        result = executor.map(task, items)
        assert result == items

    def test_map_empty_list(self):
        """Test mapping over an empty list."""
        executor = SequentialExecutor()
        task = Task(fn=lambda x: x)
        result = executor.map(task, [])
        assert result == []

    def test_map_with_scalar_and_list(self):
        """Test that scalar arguments are broadcast to match list size."""
        executor = SequentialExecutor()
        task = Task(fn=lambda db, item: f"{db}:{item}")
        result = executor.map(task, "conn", [1, 2, 3])
        assert result == ["conn:1", "conn:2", "conn:3"]

    def test_rejects_multi_cpu_task(self):
        """Test that SequentialExecutor rejects tasks requiring multiple CPUs."""
        executor = SequentialExecutor()
        task = Task(fn=lambda x: x, num_cpus=4)
        with pytest.raises(ValueError, match="Expected num_cpus to be 1 or unset"):
            executor.map(task, [1, 2])

    def test_accepts_single_cpu_task(self):
        """Test that num_cpus=1 is accepted."""
        executor = SequentialExecutor()
        task = Task(fn=lambda x: x, num_cpus=1)
        result = executor.map(task, [1, 2])
        assert result == [1, 2]

    def test_accepts_none_cpu_task(self):
        """Test that num_cpus=None (default) is accepted."""
        executor = SequentialExecutor()
        task = Task(fn=lambda x: x)
        result = executor.map(task, [1])
        assert result == [1]

    def test_submit_not_implemented(self):
        """Test that submit raises NotImplementedError."""
        executor = SequentialExecutor()
        task = Task(fn=lambda: None)
        with pytest.raises(NotImplementedError):
            executor.submit(task)

    def test_map_exception_propagates(self):
        """Test that exceptions from the mapped function propagate."""
        executor = SequentialExecutor()

        def fail(x):
            raise ValueError("boom")

        task = Task(fn=fail)
        with pytest.raises(ValueError, match="boom"):
            executor.map(task, [1])
