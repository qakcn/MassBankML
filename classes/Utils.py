import time

class Timer:
    timer = []
    named_timer={}

    @staticmethod
    def tick(name=None):
        t=time.perf_counter()
        if name is None:
            Timer.timer.append(t)
        else:
            Timer.named_timer[name]=t

    @staticmethod
    def tock(name=None):
        if name is None:
            b=Timer.timer.pop()
        else:
            b=Timer.named_timer[name]
        return time.perf_counter() - b

class Constant:
    Elements=['C', 'H', 'O', 'N', 'P', 'S', 'Cl', 'I', 'Br', 'F', 'Si', 'As']