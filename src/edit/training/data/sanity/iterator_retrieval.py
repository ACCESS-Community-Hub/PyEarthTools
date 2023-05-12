from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError

from edit.data import DataNotFoundError
from edit.training.data import DataIterator, DataStep


def _get_iterator_name(iterator: DataIterator):
    return iterator.__class__.__name__


# def _get_data(i):
#     num_samples = getattr(globals(), "samples", 5)
#     iterator = iterators[i]

#     samples = []
#     # if index:
#     #     return iterator[index]
#     for i, data in enumerate(iterator):
#         if i > num_samples:
#             break
#         samples.append(data)

#     return samples


# @functools.lru_cache(2)
# def safely_get_data(
#     iterator: DataIterator, idx: str = None, num_samples: int = 1, timeout: int = 30
# ):
#     global iterators
#     iterators = [iterator]

#     global samples
#     samples = num_samples

#     global index
#     index = idx

#     while hasattr(iterator, "iterator") and isinstance(
#         getattr(iterator, "iterator"), (DataInterface, DataIterator)
#     ):
#         iterator = iterator.iterator
#         iterators.append(iterator)
#     iterators.reverse()

#     data_samples = {}
#     with multiprocessing.Pool(processes=None) as pool:
#         it = pool.imap(
#             _get_data,
#             range(len(iterators)),
#         )
#         # it = map(_get_data, range(len(iterators)))

#         counter = 0
#         while True:
#             if counter >= len(iterators):
#                 break
#             try:
#                 #data_samples[iterators[counter]] = next(it)
#                 data_samples[iterators[counter]] = it.next(timeout=timeout)
#             except StopIteration:
#                 break
#             except multiprocessing.TimeoutError:
#                 data_samples[
#                     iterators[counter]
#                 ] = f"Data took longer than {timeout} seconds to get."

#             counter += 1

#     return data_samples



def _get_data(iterator, num_samples: int = 2, index: int = None):
    samples = None
    if index:
        return iterator[index]

    for i, data in enumerate(iterator):
        if i > num_samples:
            break
        samples = data

    return samples


# @functools.lru_cache(2)
def signal_data(
    iterator: DataStep, idx: str = None, num_samples: int = 1, timeout: int = None, show_all: bool = False
):
    iterators = [iterator]

    while hasattr(iterator, "index"):
        iterator = iterator.index
        if hasattr(iterator, "ignore_sanity") and iterator.ignore_sanity and not show_all:
            continue

        if isinstance(iterator, list):
            for item in iterator:
                iterators.append(item)
            iterator = iterator[0]
            continue

        iterators.append(iterator)

    iterators.reverse()

    data_samples = {}
    with ThreadPoolExecutor(max_workers=len(iterators)) as executor:
        futures = [_get_data(iterator, num_samples, idx) for iterator in iterators] #executor.submit
        for i, data in enumerate(futures):
            iter = iterators[i]
            try:
                data_samples[iter] = data #data.result(timeout=timeout)
            except TimeoutError:
                data_samples[iter] = f"Data took longer than {timeout} seconds to get."
            except (DataNotFoundError, RuntimeError):
                if idx is None:
                    data_samples[iter] = "Iterator likely not set, Cannot retrieve data."
                else:
                    data_samples[iter] = f"Unable to find data at {idx!r}"
    return data_samples
