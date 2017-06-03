from test_run import *
import matplotlib.pyplot as plt
import time

start_time = time.time()

x = []
y = []
z = []
a = []
r = []
f = []

for i in range(0, 11, 1):
    x.append(i/10)
    prediction, data = test('sample.csv', 0.6, 0.2, 0.2, i/10)
    acc, rec, fscore = performance(prediction, data)
    y.append(sum(prediction))
    a.append(acc)
    r.append(rec)
    f.append(fscore)
    summ = 0
    for d in data:
        summ += d[2]
    z.append(summ)

print("Run time {}".format(time.time() - start_time))

fig, ax1 = plt.subplots()
ax1.plot(x, y, 'r')
ax1.plot(x, z, 'g')

ax2 = ax1.twinx()
ax2.plot(x, a, 'b--')
ax2.plot(x, r, 'b-.')
ax2.plot(x, f, 'b:')


fig.tight_layout()
plt.show()
