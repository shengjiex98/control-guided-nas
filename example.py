from src import wrapper

latency = 0.02
errors = [0.27, 0.27] # Error in each dimension

print(wrapper.get_max_diam(latency, errors))
