from src import wrapper
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import numpy as np

latency = np.arange(0.001, 0.2, 0.001)
accuracy = np.arange(0.5, 1.0, 0.01)

def diam(latency: float, accuracy: float, sys: str):
    return wrapper.get_max_diam(latency, (1 - accuracy)**2, sys=sys)

systems = ["F1", "CC"]

diameters =[]
# Layer 11 TCP results
for l in latency:
    for a in accuracy:
        for sys in systems:
            diameters.append([l, a, sys, diam(l, a, sys=sys)])

df = pd.DataFrame(diameters, columns=["latency", "accuracy", "system", "diameter"])
print(df)

# Plotting
# for sys in ["F1", "CC"]:
#     for setup in ["l11_tcp", "l15_tcp", "l11_udp", "l15_udp"]:
#         plt.plot(range(1, 6), diameter[sys][setup], label=f"{sys} {setup}", marker="o", lw=4, markersize=15)
#     plt.xticks(range(1, 6), fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.xlabel("Network loss (%)", fontsize=20)
#     plt.ylabel("Max diameter of reachable sets (m)", fontsize=20)
#     # plt.title("Reachability analysis with a simplified cruise control model", fontsize=18)
#     plt.legend(fontsize=15)
#     plt.show()
