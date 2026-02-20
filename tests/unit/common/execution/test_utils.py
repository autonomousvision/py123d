from py123d.common.execution.sequential_executor import SequentialExecutor
from py123d.common.execution.thread_pool_executor import ThreadPoolExecutor
from py123d.common.execution.utils import (
    chunk_list,
    executor_map_chunked_list,
    executor_map_chunked_single,
    executor_map_queued,
)


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


class TestExecutorMapChunkedList:
    """Tests for executor_map_chunked_list where fn operates on chunks."""

    def test_sequential_map(self):
        """Test executor_map_chunked_list with SequentialExecutor."""
        executor = SequentialExecutor()
        result = executor_map_chunked_list(executor, lambda items: [x * 2 for x in items], [1, 2, 3])
        assert sorted(result) == [2, 4, 6]

    def test_thread_pool_map(self):
        """Test executor_map_chunked_list with ThreadPoolExecutor."""
        executor = ThreadPoolExecutor(max_workers=2)
        result = executor_map_chunked_list(executor, lambda items: [x * 2 for x in items], [1, 2, 3, 4])
        assert sorted(result) == [2, 4, 6, 8]

    def test_empty_input(self):
        """Test executor_map_chunked_list with an empty input list."""
        executor = ThreadPoolExecutor(max_workers=2)
        result = executor_map_chunked_list(executor, lambda items: [x * 2 for x in items], [])
        assert result == []

    def test_result_is_flattened(self):
        """Test that executor_map_chunked_list flattens chunked results."""
        executor = ThreadPoolExecutor(max_workers=2)
        result = executor_map_chunked_list(executor, lambda items: items, [1, 2, 3, 4, 5, 6])
        assert sorted(result) == [1, 2, 3, 4, 5, 6]

    def test_preserves_all_elements(self):
        """Test that no elements are lost during chunking and flattening."""
        executor = ThreadPoolExecutor(max_workers=4)
        items = list(range(100))
        result = executor_map_chunked_list(executor, lambda chunk: chunk, items)
        assert sorted(result) == items


class TestExecutorMapChunkedSingle:
    """Tests for executor_map_chunked_single where fn operates on individual items."""

    def test_sequential_map_single(self):
        """Test executor_map_chunked_single with SequentialExecutor."""
        executor = SequentialExecutor()
        result = executor_map_chunked_single(executor, lambda x: x * 2, [1, 2, 3])
        assert sorted(result) == [2, 4, 6]

    def test_thread_pool_map_single(self):
        """Test executor_map_chunked_single with ThreadPoolExecutor."""
        executor = ThreadPoolExecutor(max_workers=2)
        result = executor_map_chunked_single(executor, lambda x: x * 2, [1, 2, 3, 4])
        assert sorted(result) == [2, 4, 6, 8]

    def test_empty_input(self):
        """Test executor_map_chunked_single with an empty input list."""
        executor = ThreadPoolExecutor(max_workers=2)
        result = executor_map_chunked_single(executor, lambda x: x * 2, [])
        assert result == []

    def test_preserves_all_elements(self):
        """Test that no elements are lost."""
        executor = ThreadPoolExecutor(max_workers=4)
        items = list(range(100))
        result = executor_map_chunked_single(executor, lambda x: x, items)
        assert sorted(result) == items

    def test_fn_receives_single_item(self):
        """Test that fn is called with individual items, not chunks."""
        executor = ThreadPoolExecutor(max_workers=2)
        result = executor_map_chunked_single(executor, lambda x: x + 10, [1, 2, 3])
        assert sorted(result) == [11, 12, 13]


class TestExecutorMapQueued:
    """Tests for executor_map_queued where items are individually queued to workers."""

    def test_sequential_map_queued(self):
        """Test executor_map_queued with SequentialExecutor."""
        executor = SequentialExecutor()
        result = executor_map_queued(executor, lambda x: x * 2, [1, 2, 3])
        assert sorted(result) == [2, 4, 6]

    def test_thread_pool_map_queued(self):
        """Test executor_map_queued with ThreadPoolExecutor."""
        executor = ThreadPoolExecutor(max_workers=2)
        result = executor_map_queued(executor, lambda x: x * 2, [1, 2, 3, 4])
        assert sorted(result) == [2, 4, 6, 8]

    def test_empty_input(self):
        """Test executor_map_queued with an empty input list."""
        executor = ThreadPoolExecutor(max_workers=2)
        result = executor_map_queued(executor, lambda x: x * 2, [])
        assert result == []

    def test_preserves_all_elements(self):
        """Test that no elements are lost."""
        executor = ThreadPoolExecutor(max_workers=4)
        items = list(range(100))
        result = executor_map_queued(executor, lambda x: x, items)
        assert sorted(result) == items

    def test_fn_receives_single_item(self):
        """Test that fn is called with individual items, not chunks."""
        executor = ThreadPoolExecutor(max_workers=2)
        result = executor_map_queued(executor, lambda x: x + 10, [1, 2, 3])
        assert sorted(result) == [11, 12, 13]

    def test_produces_same_results_as_chunked_single(self):
        """Test that queued and chunked_single produce identical results."""
        executor = ThreadPoolExecutor(max_workers=3)
        items = list(range(50))
        fn = lambda x: x**2

        result_chunked = executor_map_chunked_single(executor, fn, items)
        result_queued = executor_map_queued(executor, fn, items)

        assert sorted(result_chunked) == sorted(result_queued)
