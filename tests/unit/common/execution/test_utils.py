from py123d.common.execution.sequential_executor import SequentialExecutor
from py123d.common.execution.thread_pool_executor import ThreadPoolExecutor
from py123d.common.execution.utils import chunk_list, executor_map


class TestChunkList:
    """Tests for the chunk_list function to ensure it correctly splits lists into chunks."""

    def test_basic_chunking(self):
        """Test splitting a list into the specified number of chunks."""
        chunks = chunk_list([1, 2, 3, 4, 5, 6], num_chunks=3)
        assert len(chunks) == 3
        # All elements present
        flattened = [item for chunk in chunks for item in chunk]
        assert sorted(flattened) == [1, 2, 3, 4, 5, 6]

    def test_uneven_split(self):
        """Test that uneven splits distribute elements correctly."""
        chunks = chunk_list([1, 2, 3, 4, 5], num_chunks=3)
        assert len(chunks) == 3
        flattened = [item for chunk in chunks for item in chunk]
        assert sorted(flattened) == [1, 2, 3, 4, 5]

    def test_more_chunks_than_items(self):
        """Test that empty chunks are filtered out."""
        chunks = chunk_list([1, 2], num_chunks=5)
        assert len(chunks) == 2
        flattened = [item for chunk in chunks for item in chunk]
        assert sorted(flattened) == [1, 2]

    def test_single_chunk(self):
        """Test splitting into a single chunk."""
        chunks = chunk_list([1, 2, 3], num_chunks=1)
        assert len(chunks) == 1
        assert chunks[0] == [1, 2, 3]

    def test_empty_list(self):
        """Test chunking an empty list."""
        chunks = chunk_list([], num_chunks=3)
        assert chunks == []

    def test_default_num_chunks(self):
        """Test that default num_chunks produces at least 1 chunk."""
        chunks = chunk_list([1, 2, 3])
        assert len(chunks) >= 1
        flattened = [item for chunk in chunks for item in chunk]
        assert sorted(flattened) == [1, 2, 3]


class TestExecutorMap:
    """Various tests for the executor_map function with different executors and input scenarios."""

    def test_sequential_map(self):
        """Test executor_map with SequentialExecutor."""
        executor = SequentialExecutor()
        result = executor_map(executor, lambda items: [x * 2 for x in items], [1, 2, 3])
        assert sorted(result) == [2, 4, 6]

    def test_thread_pool_map(self):
        """Test executor_map with ThreadPoolExecutor."""
        executor = ThreadPoolExecutor(max_workers=2)
        result = executor_map(executor, lambda items: [x * 2 for x in items], [1, 2, 3, 4])
        assert sorted(result) == [2, 4, 6, 8]

    def test_empty_input(self):
        """Test executor_map with an empty input list."""
        executor = ThreadPoolExecutor(max_workers=2)
        result = executor_map(executor, lambda items: [x * 2 for x in items], [])
        assert result == []

    def test_result_is_flattened(self):
        """Test that executor_map flattens chunked results."""
        executor = ThreadPoolExecutor(max_workers=2)
        # The function receives a chunk and returns a list; executor_map should flatten.
        result = executor_map(executor, lambda items: items, [1, 2, 3, 4, 5, 6])
        assert sorted(result) == [1, 2, 3, 4, 5, 6]

    def test_preserves_all_elements(self):
        """Test that no elements are lost during chunking and flattening."""
        executor = ThreadPoolExecutor(max_workers=4)
        items = list(range(100))
        result = executor_map(executor, lambda chunk: chunk, items)
        assert sorted(result) == items
