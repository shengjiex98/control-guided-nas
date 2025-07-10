from control_guided_nas import get_max_diam

# Single dimension sensing example
print(get_max_diam(0.02, 0.2, sysname="F1"))

# Multi dimension sensing example
print(get_max_diam(0.02, [0.2, 0.1, 0.05], sysname="ACCLK"))
