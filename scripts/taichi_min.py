import numpy as np
import taichi as ti

ti.init(arch=ti.cuda)
N = 10000000
K = 2
arr = np.random.randn(N, K).astype(np.float32)
ti_arr = ti.Vector(2, dt=ti.f32, shape=(N,))
ti_arr.from_numpy(arr)


@ti.kernel
def reduce_min() -> ti.f32:
    min_val = float("inf")
    smallest = ti.Vector([0.0, 0.0], dt=ti.f32)
    for i in ti_arr:
        if ti_arr[i].norm() < min_val:
            min_val = ti_arr[i].norm()
            smallest = ti_arr[i]
    print(smallest[0])
    print(smallest[1])
    return min_val


if __name__ == "__main__":

    norms = np.linalg.norm(arr, axis=-1)
    print(min(norms), arr[np.argmin(norms)])
    print(reduce_min())
