import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""
xlabel = ["overall", "128~183", "184~255", "256~383", "384~511", "512~"]
x1 = np.arange(6, dtype=float)
y1 = [0.6134, 0.69230769, 0.63416305, 0.62415631, 0.48876175, 0.48154093]
y2 = [0.6097, 0.68817204, 0.62426682, 0.62619069, 0.49574727, 0.49152542]
y3 = [0.5856486, 0.6743, 0.607144, 0.583605, 0.459949, 0.400000]
"""
"""
xlabel = ["overall", "1~10", "11~15", "16~20", "21~25", "26~30", "31~35", "35~"]
x1 = np.arange(8, dtype=float)
y1 = [0.6134, 0.69208211, 0.63309353, 0.6150007, 0.61188715, 0.59113091, 0.5883333, 0.47933884]
y2 = [0.6097, 0.68468468, 0.63721941, 0.61025943, 0.60140826, 0.60525201, 0.58959538, 0.40677966]
y3 = [0.5856486, 0.674444, 0.627922, 0.602993, 0.584318, 0.564229, 0.5442684, 0.38855]

plt.bar(x1 - 0.25, y1, width=0.25, label="graphfusion_implicit",)
plt.bar(x1, y2, width=0.25, label="graphfusion_explicit",)
plt.bar(x1 + 0.25, y3, width=0.25, label="bert_base")
plt.xticks(x1, xlabel,)
plt.xlabel("number of entities")
plt.ylabel("F1 score")
plt.legend()
plt.show()
"""

x = [2, 4, 6, 8, 10, 12]
y0 = [59.29, 59.29, 59.29, 59.29, 59.29, 59.29]
y1 = [60.14, 59.01, 51.24, 30.03, 12.57, 4.31]
y2 = [60.92, 60.99, 61.13, 61.23, 61.29, 61.34]
y3 = [60.97, 60.47, 59.06, 57.92, 54.51, 50.97]
"""
plt.plot(x, y0, label="plm", linewidth=2, color="grey")
plt.plot(x, y2, label="implicit", marker="^", linewidth=2)
plt.xlabel("k")
plt.ylabel("F1 score")
"""
plt.plot(x, y0, label="plm", linewidth=2, color="grey")
plt.plot(x, y1, label="gnn", marker="x", linewidth=2)
plt.plot(x, y3, label="explicit", marker="o", linewidth=2)
plt.xlabel("k")
plt.ylabel("F1 score")

plt.legend()
plt.show()

"""
att_value = np.array([[23010, 24378, 23869, 28912, 21955,],
    [42805, 53932, 34414, 55982, 42071,],
    [48237, 52726, 55536, 72633, 44144,],
    [55290, 58134, 56521, 70211, 49052,],
    [63860, 98242, 92158, 78479, 74409,],
    [48850, 105141, 72668, 60177, 56798,],
    [63250, 137551, 103961, 63535, 61500,],
    [89837, 125584, 79701, 141030, 94906,],
    [69879, 118432, 47738, 105993, 77824,],
    [78385, 133321, 61434, 149343, 88878,],
    [59098, 163293, 93061, 112219, 76887,],
    [64407, 110503, 153015, 103717, 73708,],
]).transpose()
att_value =  att_value * 1.0 / 100000
xlabel = np.arange(12) + 1
ylabel = ["MMO", "MMR", "MEA", "EEC", "SSC",]
sns.heatmap(att_value, cmap="GnBu", annot=True)
plt.xticks(np.arange(12) + 0.5, xlabel)
plt.yticks(np.arange(5) + 0.5, ylabel, rotation=0)
plt.legend()
plt.show()
"""