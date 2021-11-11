import ray
import numpy as np
import time

class inst(object):
    def __init__(self):
        self.id=1

    def compute(self):
        time.sleep(0.1)
        for i in range(1000):
            self.vals=np.random.random(100)
        self.id = np.random.normal(size=5000000)


class sim(object):
    def __init__(self):
        self.inst=[inst() for _ in range(20)]
        self.id=1

    @ray.remote
    def compute(self,item):
        item[0]=203
        return item


ray.init()

z=sim()

ids=[ray.put([1,2,3,4]) for _ in range(20)]

t1=time.time()
ray.get([z.compute.remote(z,idx) for idx in ids])


print([ray.get(idx) for idx in ids])

print(time.time()-t1,z.id)


