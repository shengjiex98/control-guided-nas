from src import wrapper
import matplotlib.pyplot as plt
from itertools import product

latency = {
    "l11_tcp": [0.006, 0.018, 0.029, 0.051, 0.066],
    "l15_tcp": [0.003, 0.005, 0.011, 0.013, 0.021],
    "l11_udp": [0.006],
    "l15_udp": [0.005]
}

accuracy = {
    "l11_tcp": [0.6501],
    "l15_tcp": [0.6756],
    "l11_udp": [0.6198, 0.5807, 0.5477, 0.5193, 0.4888],
    "l15_udp": [0.6453, 0.6062, 0.5732, 0.5448, 0.5143]
}

diameter = {}

def diam(latency: list[float], accuracy: list[float], sys: str):
    return [wrapper.get_max_diam(l, (1 - a)**2, sys=sys) for l, a in product(latency, accuracy)]

# Layer 11 TCP results
for setup in ["l11_tcp", "l15_tcp", "l11_udp", "l15_udp"]:
    diameter[setup] = diam(latency[setup], accuracy[setup], sys="F1")
    print(f"{setup} results")
    print(diameter[setup])

# Plotting
for setup in ["l11_tcp", "l15_tcp", "l11_udp", "l15_udp"]:
    plt.plot(range(1, 6), diameter[setup], label=setup, marker="x", lw=1)
plt.xticks(range(1, 6))
plt.xlabel("Packet loss (%)")
plt.ylabel("Maximal diameter of the reachable sets (m)")
plt.title("Reachability analysis with a simplified F1/10 car model")
plt.legend()
plt.show()
