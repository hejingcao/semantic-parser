# -*- coding: utf-8 -*-

import multiprocessing
import time


class TimeoutException(Exception):
    """ It took too long to compile and execute. """


class RunnableProcessing(multiprocessing.Process):
    """ Run a function in a child process.

    Pass back any exception received.
    """

    def __init__(self, func, *args, **kwargs):
        self.queue = multiprocessing.Queue(maxsize=1)
        args = (func,) + args
        multiprocessing.Process.__init__(self, target=self.run_func, args=args, kwargs=kwargs)

    def run_func(self, func, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
            self.queue.put((True, result))
        except Exception as e:
            self.queue.put((False, e))

    def done(self):
        return self.queue.full()

    def result(self):
        return self.queue.get()


def timeout(seconds, force_kill=True):
    """ Timeout decorator using Python multiprocessing.

    Courtesy of http://code.activestate.com/recipes/577853-timeout-decorator-with-multiprocessing/
    """
    def wrapper(function):
        def inner(*args, **kwargs):
            now = time.time()
            proc = RunnableProcessing(function, *args, **kwargs)
            proc.start()
            proc.join(seconds)
            if proc.is_alive():
                if force_kill:
                    proc.terminate()
                runtime = time.time() - now
                raise TimeoutException('timed out after {0} seconds'.format(runtime))
            assert proc.done()
            success, result = proc.result()
            if success:
                return result
            else:
                raise result
        return inner
    return wrapper
