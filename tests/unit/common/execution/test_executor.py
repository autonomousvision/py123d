import pytest

from py123d.common.execution.executor import (
    Executor,
    ExecutorResources,
    Task,
    align_size_of_arguments,
    get_max_size_of_arguments,
)


class TestTask:
    def test_call(self):
        """Test that Task delegates to the wrapped function."""
        task = Task(fn=lambda x, y: x + y)
        assert task(2, 3) == 5

    def test_call_with_kwargs(self):
        """Test that Task passes keyword arguments."""
        task = Task(fn=lambda x, y=10: x + y)
        assert task(1, y=5) == 6

    def test_default_resources(self):
        """Test that default resource requirements are None."""
        task = Task(fn=lambda: None)
        assert task.num_cpus is None
        assert task.num_gpus is None

    def test_custom_resources(self):
        """Test task with explicit resource requirements."""
        task = Task(fn=lambda: None, num_cpus=4, num_gpus=0.5)
        assert task.num_cpus == 4
        assert task.num_gpus == 0.5

    def test_frozen(self):
        """Test that Task is immutable."""
        task = Task(fn=lambda: None)
        with pytest.raises(AttributeError):
            task.num_cpus = 2  # type: ignore


class TestExecutorResources:
    def test_number_of_threads_single_node(self):
        """Test thread count for a single node."""
        resources = ExecutorResources(number_of_nodes=1, number_of_cpus_per_node=8, number_of_gpus_per_node=0)
        assert resources.number_of_threads == 8

    def test_number_of_threads_multi_node(self):
        """Test thread count across multiple nodes."""
        resources = ExecutorResources(number_of_nodes=3, number_of_cpus_per_node=4, number_of_gpus_per_node=2)
        assert resources.number_of_threads == 12

    def test_current_node_cpu_count(self):
        """Test that current_node_cpu_count returns a positive integer."""
        count = ExecutorResources.current_node_cpu_count()
        assert isinstance(count, int)
        assert count >= 1

    def test_frozen(self):
        """Test that ExecutorResources is immutable."""
        resources = ExecutorResources(number_of_nodes=1, number_of_cpus_per_node=1, number_of_gpus_per_node=0)
        with pytest.raises(AttributeError):
            resources.number_of_nodes = 2  # type: ignore


class TestExecutor:
    def test_cannot_instantiate_abstract(self):
        """Test that Executor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Executor(ExecutorResources(number_of_nodes=1, number_of_cpus_per_node=1, number_of_gpus_per_node=0))  # type: ignore

    def test_rejects_zero_threads(self):
        """Test that Executor raises on zero threads."""

        class DummyExecutor(Executor):
            def _map(self, task, *item_lists, verbose=False):
                return []

            def submit(self, task, *args, **kwargs):
                raise NotImplementedError

        with pytest.raises(RuntimeError, match="Number of threads can not be 0"):
            DummyExecutor(ExecutorResources(number_of_nodes=0, number_of_cpus_per_node=1, number_of_gpus_per_node=0))

    def test_str(self):
        """Test string representation of an Executor."""

        class DummyExecutor(Executor):
            def _map(self, task, *item_lists, verbose=False):
                return []

            def submit(self, task, *args, **kwargs):
                raise NotImplementedError

        executor = DummyExecutor(
            ExecutorResources(number_of_nodes=2, number_of_cpus_per_node=4, number_of_gpus_per_node=1)
        )
        s = str(executor)
        assert "Number of nodes: 2" in s
        assert "Number of CPUs per node: 4" in s
        assert "Number of GPUs per node: 1" in s
        assert "Number of threads across all nodes: 8" in s

    def test_number_of_threads_property(self):
        """Test that executor exposes number_of_threads from config."""

        class DummyExecutor(Executor):
            def _map(self, task, *item_lists, verbose=False):
                return []

            def submit(self, task, *args, **kwargs):
                raise NotImplementedError

        executor = DummyExecutor(
            ExecutorResources(number_of_nodes=1, number_of_cpus_per_node=16, number_of_gpus_per_node=0)
        )
        assert executor.number_of_threads == 16


class TestGetMaxSizeOfArguments:
    def test_single_list(self):
        """Test with a single list argument."""
        assert get_max_size_of_arguments([1, 2, 3]) == 3

    def test_multiple_same_size_lists(self):
        """Test with multiple lists of equal size."""
        assert get_max_size_of_arguments([1, 2], [3, 4]) == 2

    def test_list_and_scalar(self):
        """Test with a list and a non-list argument."""
        assert get_max_size_of_arguments("scalar", [1, 2, 3]) == 3

    def test_no_lists(self):
        """Test with only non-list arguments returns 1."""
        assert get_max_size_of_arguments("a", "b") == 1

    def test_mismatched_list_sizes_raises(self):
        """Test that differently-sized lists raise an error."""
        with pytest.raises(RuntimeError, match="different element size"):
            get_max_size_of_arguments([1, 2], [3, 4, 5])


class TestAlignSizeOfArguments:
    def test_scalar_is_repeated(self):
        """Test that non-list arguments are repeated to match list size."""
        max_size, aligned = align_size_of_arguments("db", [1, 2, 3])
        assert max_size == 3
        assert aligned[0] == ["db", "db", "db"]
        assert aligned[1] == [1, 2, 3]

    def test_lists_unchanged(self):
        """Test that equal-sized lists pass through unchanged."""
        max_size, aligned = align_size_of_arguments([1, 2], [3, 4])
        assert max_size == 2
        assert aligned[0] == [1, 2]
        assert aligned[1] == [3, 4]
