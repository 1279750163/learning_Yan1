import numpy as np

# a = np.array([[1, 2], [1, 2], [2, 2]])
# print(np.average(a, axis=0))
#
# a = {1: [1, 2], 7: [2, 3]}
# for i in a:
#     print(i)
# b = dict(a)
# b[1] = 2
# print(b)
# print(a)
a = [1, 2]
b = [1, 0]
m = list(map(lambda x: a[x], b))
print(m)
x =list(map(lambda x: x*x,[y for y in range(10)]))
print(x)