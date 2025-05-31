import numpy as np
import matplotlib.pyplot as plt


def half_moon(n, theta=np.pi):
    t = np.random.rand(n)
    x = np.cos(t * theta)
    y = np.sin(t * theta)
    return np.stack([x, y], axis=-1)


def half_moons(n, *, k=5, offset=3.0):
    
    def rot2d(t):
        sin = np.sin(t)
        cos = np.cos(t)
        return np.array([[cos, -sin], [sin, cos]])
    
    result = np.empty([0, 2])
    for i in range(k):
        result = np.einsum('ij, ni -> nj', rot2d(2 * np.pi / k), result)
        
        xs = half_moon(n // k, theta=np.pi)
        xs = xs - offset
        xs = np.einsum('ij, ni -> nj', rot2d(np.pi / 5), xs)
        xs = xs + offset
        
        result = np.concatenate([result, xs], axis=0)
    return result


def randn_like(xs):
    return np.random.randn(*xs.shape)


def show_2d(xs, **kwargs):
    plt.scatter(xs[:,0], xs[:,1], **kwargs)
    plt.grid(True)
    plt.axis('equal')