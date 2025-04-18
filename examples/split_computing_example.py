import matplotlib.pyplot as plt

from control_guided_nas import get_max_diam

latency = {
    "l11_tcp": [0.006, 0.018, 0.029, 0.051, 0.066],
    "l11_udp": [0.006, 0.006, 0.006, 0.006, 0.006],
    "l15_tcp": [0.003, 0.005, 0.011, 0.013, 0.021],
    "l15_udp": [0.005, 0.005, 0.005, 0.005, 0.005]
}

accuracy = {
    "l11_tcp": [0.6501, 0.6501, 0.6501, 0.6501, 0.6501],
    "l11_udp": [0.6198, 0.5807, 0.5477, 0.5193, 0.4888],
    "l15_tcp": [0.6756, 0.6756, 0.6756, 0.6756, 0.6756],
    "l15_udp": [0.6453, 0.6062, 0.5732, 0.5448, 0.5143]
}

diameter = {
    "F1": {},
    "CC": {}
}

def diam(latency: list[float], accuracy: list[float], sys: str):
    return [get_max_diam(l, (1 - a)**2, sysname=sys) for l, a in zip(latency, accuracy)]

for sys in ["F1", "CC"]:
    for setup in latency.keys():
        diameter[sys][setup] = diam(latency[setup], accuracy[setup], sys=sys)
        print(f"{sys} {setup} results")
        print(diameter[sys][setup])

colors = {"l11": "lightblue", "l15": "lightpink"}
lstyle = {"l11": "solid", "l15": "dashed"}
marker = {"tcp": "o", "udp": "^"}

# plt.style.use('ggplot')

for val, y in zip([latency, accuracy], ["Latency (sec)", "Accuracy"]):
    for setup in val.keys():
        layer, proto = setup.split('_')
        plt.plot(range(1, 6), val[setup], label=f"{setup}", color=colors[layer], marker=marker[proto], linestyle=lstyle[layer], lw=2, markersize=15)
    plt.xticks(range(1, 6), fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Network loss (%)", fontsize=22)
    plt.ylabel(y, fontsize=22)
    # plt.title("Reachability analysis with a simplified cruise control model", fontsize=18)
    plt.legend(fontsize=18)
    plt.show()

# Plotting
for sys in ["F1", "CC"]:
    for setup in latency.keys():
        layer, proto = setup.split('_')
        plt.plot(range(1, 6), diameter[sys][setup], label=f"{setup}", color=colors[layer], marker=marker[proto], linestyle=lstyle[layer], lw=2, markersize=15)
    plt.xticks(range(1, 6), fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Network loss (%)", fontsize=22)
    plt.ylabel("Max diameter of reachable sets (m)", fontsize=22)
    # plt.title("Reachability analysis with a simplified cruise control model", fontsize=18)
    plt.legend(fontsize=18)
    plt.show()
