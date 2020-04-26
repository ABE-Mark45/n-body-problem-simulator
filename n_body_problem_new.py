import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')
from scipy.integrate import odeint
from itertools import cycle

G = 1

def rand(a, b):
    return np.random.uniform(a, b)

r1 = np.array([0.97000436, -0.24308753])
r2 = -1*r1
r3 = np.array([0.0, 0.0])

R = [r1, r2, r3]

nBodies = len(R)

vel = 2
v3 = np.array([-0.93240737, -0.86473146])
v1 = -0.5*v3
v2 = -0.5*v3


V = [v1, v2, v3]

M = np.array([1, 1, 1])

def newderivatives(inputVec, t):
    n = len(inputVec) // 2
    r = inputVec[:n]
    v = inputVec[n:]

    dvbydt = np.zeros_like(v)
    drbydt = np.zeros_like(r)
    for i in range(n//2):
        r1 = r[2*i: 2*i+2]
        v1 = v[2*i: 2*i+2]

        dv1bydt = np.array([0.0, 0.0])
        dr1bydt = v1
        for j in range(n//2):
            if i == j:
                continue
            r2 = r[2*j: 2*j+2]
            if np.linalg.norm(r2 - r1) < 0.0001:
                dr1bydt = v[2*j: 2*j+2]
                continue
            dv1bydt += G * M[j] * np.array(r2 - r1) / np.linalg.norm(r2-r1)**3

        dvbydt[2*i: 2*i+2] = dv1bydt
        drbydt[2*i: 2*i+2] = dr1bydt
    return np.array([drbydt, dvbydt]).flatten()

initial_conditions = np.array([R, V]).flatten()

numSeconds = 100
numFrames = 1800
timespan = np.linspace(0, 20, numFrames)

system_sol = odeint(newderivatives, initial_conditions, timespan)


r_sol = system_sol[:, :len(R)*2]


fig = plt.figure()
ax = plt.axes(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))

lines = []
dots = []


cycol = cycle('bgrcmk')

for index in range(nBodies):
    c = next(cycol)
    lineObj = ax.plot([], [], lw=2, color=c)[0]
    lines.append(lineObj)
    dots.append(ax.plot([], [], 'o', color=c)[0])


def init():
    for line in lines:
        line.set_data([], [])
    for dot in dots:
        dot.set_data([], [])
    return lines + dots


def animate(i):
    n = 0
    p = 150
    if i > p:
        n = i - p
    for index in range(nBodies):
        lines[index].set_data(r_sol[n:i, 2*index], r_sol[n:i, 2*index+1])
        dots[index].set_data(r_sol[i, 2*index], r_sol[i, 2*index+1])
    return tuple(lines+dots,)


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=numFrames, interval=1, blit=True)

anim.save('mobius.mp4', fps=60, extra_args=['-vcodec', 'libx264'])
plt.grid()
plt.show()
