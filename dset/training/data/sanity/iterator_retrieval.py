import functools
import multiprocessing
import signal

from dset.training.data import DataOperation, DataIterator, DataInterface


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


class TimeoutException(Exception):
    pass

def timeout_handler(num, stack):
    raise TimeoutException("Timeout Exception was Triggered")


def _get_signal_data(iterator, num_samples: int = 2, index: int = None):

    samples = []
    if index:
        yield iterator[index]
        return
    for i, data in enumerate(iterator):
        if i > num_samples:
            break
        samples.append(data)

    yield samples

#@functools.lru_cache(2)
def signal_data(
    iterator: DataIterator, idx: str = None, num_samples: int = 1, timeout: int = 30
):
    iterators = [iterator]

    while hasattr(iterator, "index"):
        iterator = iterator.index
        iterators.append(iterator)
    iterators.reverse()

    data_samples = {}
    timeouts = 0

    for iter in iterators:
        try:
            #signal.signal(signal.SIGALRM, timeout_handler)
            #signal.alarm(timeout)
            data_samples[iter] = _get_signal_data(iter, num_samples=num_samples, index=idx)
        except TimeoutException:
            timeouts += 1
            data_samples[iter] = f"Data took longer than {timeout} seconds to get."
    return data_samples
