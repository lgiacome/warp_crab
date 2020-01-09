import multiprocessing as mp


def run_in_separate_process(function, args=[], kwargs={}):

    def target_function(queue, aa, kk):
        rr = function(*aa, **kk)
        queue.put(rr)

    qq = mp.Queue()

    proc = mp.Process(target=target_function,
            args=(qq, args, kwargs))

    proc.start()

    proc.join()

    res = qq.get()

    return res
