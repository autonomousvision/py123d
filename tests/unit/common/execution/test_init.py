"""Test that the execution module exports all public names correctly."""


class TestModuleExports:
    def test_executor_importable(self):
        from py123d.common.execution import Executor

        assert Executor is not None

    def test_executor_resources_importable(self):
        from py123d.common.execution import ExecutorResources

        assert ExecutorResources is not None

    def test_task_importable(self):
        from py123d.common.execution import Task

        assert Task is not None

    def test_sequential_executor_importable(self):
        from py123d.common.execution import SequentialExecutor

        assert SequentialExecutor is not None

    def test_thread_pool_executor_importable(self):
        from py123d.common.execution import ThreadPoolExecutor

        assert ThreadPoolExecutor is not None

    def test_process_pool_executor_importable(self):
        from py123d.common.execution import ProcessPoolExecutor

        assert ProcessPoolExecutor is not None

    def test_executor_map_importable(self):
        from py123d.common.execution import executor_map

        assert executor_map is not None

    def test_chunk_list_importable(self):
        from py123d.common.execution import chunk_list

        assert chunk_list is not None

    def test_all_executors_inherit_from_base(self):
        """Test that all executor classes are subclasses of Executor."""
        from py123d.common.execution import (
            Executor,
            ProcessPoolExecutor,
            SequentialExecutor,
            ThreadPoolExecutor,
        )

        assert issubclass(SequentialExecutor, Executor)
        assert issubclass(ThreadPoolExecutor, Executor)
        assert issubclass(ProcessPoolExecutor, Executor)
