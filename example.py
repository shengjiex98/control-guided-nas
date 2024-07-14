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

diameter = {
    "F1": {},
    "CC": {}
}

def diam(latency: list[float], accuracy: list[float], sys: str):
    return [wrapper.get_max_diam(l, (1 - a)**2, sys=sys) for l, a in product(latency, accuracy)]

# Layer 11 TCP results
for sys in ["F1", "CC"]:
    for setup in ["l11_tcp", "l15_tcp", "l11_udp", "l15_udp"]:
        diameter[sys][setup] = diam(latency[setup], accuracy[setup], sys=sys)
        print(f"{sys} {setup} results")
        print(diameter[sys][setup])

# Plotting
for sys in ["F1", "CC"]:
    for setup in ["l11_tcp", "l15_tcp", "l11_udp", "l15_udp"]:
        plt.plot(range(1, 6), diameter[sys][setup], label=f"{sys} {setup}", marker="o", lw=4, markersize=15)
    plt.xticks(range(1, 6), fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Network loss (%)", fontsize=20)
    plt.ylabel("Max diameter of reachable sets (m)", fontsize=20)
    # plt.title("Reachability analysis with a simplified cruise control model", fontsize=18)
    plt.legend(fontsize=15)
    plt.show()
