import multiprocessing as mp


def run_in_separate_process(function, args=[]):

    def target_function(queue, aa):
        rr = function(*aa)
        queue.put(rr)

    qq = mp.Queue()

    proc = mp.Process(target=target_function,
            args=(qq, args))

    proc.start()

    proc.join()

    res = qq.get()

    return res
