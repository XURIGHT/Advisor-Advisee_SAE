# -*-coding:utf-8-*-
import matplotlib.pyplot as plt

# fig 4.1

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.subplot(221)
plt.ylim(0.7, 0.85)
plt.title("Accuracy")

name_list = ['With content', 'Without content']
num_list = [0.778242, 0.759029]
num_list1 = [0.795116, 0.748862]
num_list2 = [0.791782, 0.759016]
num_list3 = [0.799242, 0.746286]
num_list4 = [0.793436, 0.754752]
x = list(range(len(num_list)))
total_width, n = 0.3, 2
width = total_width / n

plt.bar(x, num_list, width=width, label='LR', fc='blue')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label='SVM-rbf', fc='red')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, width=width, label='SVM-liner', tick_label=name_list, fc='green')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list3, width=width, label='XGBoost', fc='orange')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list4, width=width, label='RF', fc='yellow')

plt.subplot(222)
plt.ylim(0.5, 0.8)
plt.title("Precision")

name_list = ['With content', 'Without content']
num_list = [0.610256, 0.611872]
num_list1 = [0.661111, 0.589861]
num_list2 = [0.641026, 0.589372]
num_list3 = [0.668421, 0.587678]
num_list4 = [0.678392, 0.615741]
x = list(range(len(num_list)))
total_width, n = 0.3, 2
width = total_width / n

plt.bar(x, num_list, width=width, label='LR', fc='blue')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label='SVM-rbf', fc='red')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, width=width, label='SVM-liner', tick_label=name_list, fc='green')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list3, width=width, label='XGBoost', fc='orange')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list4, width=width, label='RF', fc='yellow')

plt.subplot(223)
plt.ylim(0.6, 0.85)
plt.title("Recall")

name_list = ['With content', 'Without content']
num_list = [0.782894, 0.797619]
num_list1 = [0.772727, 0.785276]
num_list2 = [0.791139, 0.782051]
num_list3 = [0.779141, 0.765432]
num_list4 = [0.780347, 0.767778]
x = list(range(len(num_list)))
total_width, n = 0.3, 2
width = total_width / n

plt.bar(x, num_list, width=width, label='LR', fc='blue')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label='SVM-rbf', fc='red')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, width=width, label='SVM-liner', tick_label=name_list, fc='green')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list3, width=width, label='XGBoost', fc='orange')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list4, width=width, label='RF', fc='yellow')

plt.subplot(224)
plt.ylim(0.6, 0.8)
plt.title("F1-score")

name_list = ['With content', 'Without content']
num_list = [0.685878, 0.692506]
num_list1 = [0.712574, 0.673684]
num_list2 = [0.708215, 0.672176]
num_list3 = [0.719547, 0.664879]
num_list4 = [0.725806, 0.687339]
x = list(range(len(num_list)))
total_width, n = 0.3, 2
width = total_width / n

plt.bar(x, num_list, width=width, label='LR', fc='blue')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label='SVM-rbf', fc='red')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, width=width, label='SVM-liner', tick_label=name_list, fc='green')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list3, width=width, label='XGBoost', fc='orange')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list4, width=width, label='RF', fc='yellow')

plt.legend(bbox_to_anchor=(1.3, 2.9))
plt.show()
'''
plt.ylim(0.7, 1.0)
plt.title("Comparison of using different methods")

name_list = ['With content information', 'Without content information', 'Machine learning method']
num_list = [0.8457755, 0.8222983, 0.8051162]
x = list(range(len(num_list)))
total_width, n = 0.8, 3
width = total_width / n


plt.bar(x, num_list, width=width, tick_label=name_list, color=['r', 'g', 'b'])

plt.show()
'''
# fig 4.2

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.subplot(221)
plt.ylim(0.7, 0.8)
plt.title("Accuracy")

x_data = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
y_data =  [0.751249, 0.765931, 0.772832, 0.767850, 0.768950, 0.785318, 0.778111, 0.771392, 0.789393]
y_data2 = [0.765693, 0.775471, 0.779886, 0.776085, 0.778041, 0.778077, 0.767872, 0.776908, 0.798636]
y_data3 = [0.752741, 0.763409, 0.771230, 0.768596, 0.767170, 0.785337, 0.786816, 0.779257, 0.789292]
y_data4 = [0.766661, 0.784449, 0.793358, 0.782828, 0.784313, 0.789806, 0.784053, 0.780167, 0.798212]
y_data5 = [0.743234, 0.760619, 0.769634, 0.760380, 0.761304, 0.771941, 0.767430, 0.772402, 0.766353]

plt.plot(x_data, y_data, color='red', linewidth=1.0, linestyle='-', label='LR')
plt.plot(x_data, y_data2, color='blue', linewidth=1.0, linestyle=':', label='SVM-rbf')
plt.plot(x_data, y_data3, color='green', linewidth=1.0, linestyle='--', label='SVM-liner')
plt.plot(x_data, y_data4, color='purple', linewidth=1.0, linestyle='-.', label='XGBoost')
plt.plot(x_data, y_data5, color='black', linewidth=1.0, linestyle=(0, (3, 1, 1, 1, 1, 1)), label='RF')

plt.subplot(222)
plt.ylim(0.55, 0.8)
plt.title("Precision")

y_data = [0.569307, 0.661616, 0.615023, 0.640244, 0.625698, 0.631068, 0.625616, 0.627219, 0.668269]
y_data2 = [0.625, 0.688776, 0.658065, 0.655556, 0.637931, 0.707792, 0.595745, 0.666667, 0.636842]
y_data3 = [0.582915, 0.656085, 0.664804, 0.658824, 0.65625, 0.648936, 0.672897, 0.684211, 0.678414]
y_data4 = [0.641711, 0.692683, 0.684492, 0.68254, 0.672489, 0.685039, 0.698925, 0.626506, 0.684932]
y_data5 = [0.583333, 0.62037, 0.642458, 0.663158, 0.655172, 0.644172, 0.670659, 0.696629, 0.637931]

plt.plot(x_data, y_data, color='red', linewidth=1.0, linestyle='-', label='LR')
plt.plot(x_data, y_data2, color='blue', linewidth=1.0, linestyle=':', label='SVM-rbf')
plt.plot(x_data, y_data3, color='green', linewidth=1.0, linestyle='--', label='SVM-liner')
plt.plot(x_data, y_data4, color='purple', linewidth=1.0, linestyle='-.', label='XGBoost')
plt.plot(x_data, y_data5, color='black', linewidth=1.0, linestyle=(0, (3, 1, 1, 1, 1, 1)), label='RF')

plt.subplot(223)
plt.ylim(0.6, 0.85)
plt.title("Recall")

y_data = [0.766667, 0.731844, 0.813665, 0.65625, 0.704403, 0.8125, 0.79375, 0.679487, 0.798851]
y_data2 = [0.778443, 0.72973, 0.649682, 0.710843, 0.707006, 0.630058, 0.746667, 0.764045, 0.801325]
y_data3 = [0.748387, 0.708571, 0.69186, 0.666667, 0.794595, 0.753086, 0.804469, 0.726257, 0.832432]
y_data4 = [0.714286, 0.767568, 0.748538, 0.732955, 0.832432, 0.63, 0.722222, 0.693333, 0.833333]
y_data5 = [0.673077, 0.788235, 0.70122, 0.7, 0.738889, 0.660377, 0.654971, 0.681319, 0.826816]

plt.plot(x_data, y_data, color='red', linewidth=1.0, linestyle='-', label='LR')
plt.plot(x_data, y_data2, color='blue', linewidth=1.0, linestyle=':', label='SVM-rbf')
plt.plot(x_data, y_data3, color='green', linewidth=1.0, linestyle='--', label='SVM-liner')
plt.plot(x_data, y_data4, color='purple', linewidth=1.0, linestyle='-.', label='XGBoost')
plt.plot(x_data, y_data5, color='black', linewidth=1.0, linestyle=(0, (3, 1, 1, 1, 1, 1)), label='RF')


plt.subplot(224)
plt.ylim(0.6, 0.8)
plt.title("F1-score")

y_data = [0.653409, 0.69496, 0.700535, 0.648148, 0.662722, 0.710383, 0.699725, 0.652308, 0.727749]
y_data2 = [0.693333, 0.708661, 0.653846, 0.682081, 0.670695, 0.666667, 0.662722, 0.712042, 0.709677]
y_data3 = [0.655367, 0.681319, 0.678063, 0.662722, 0.718826, 0.697143, 0.732824, 0.704607, 0.747573]
y_data4 = [0.676056, 0.728205, 0.715084, 0.706849, 0.743961, 0.628159, 0.710383, 0.658228, 0.75188]
y_data5 = [0.625, 0.694301, 0.670554, 0.681081, 0.694517, 0.652174, 0.662722, 0.688889, 0.720195]

plt.plot(x_data, y_data, color='red', linewidth=1.0, linestyle='-', label='LR')
plt.plot(x_data, y_data2, color='blue', linewidth=1.0, linestyle=':', label='SVM-rbf')
plt.plot(x_data, y_data3, color='green', linewidth=1.0, linestyle='--', label='SVM-liner')
plt.plot(x_data, y_data4, color='purple', linewidth=1.0, linestyle='-.', label='XGBoost')
plt.plot(x_data, y_data5, color='black', linewidth=1.0, linestyle=(0, (3, 1, 1, 1, 1, 1)), label='RF')

plt.legend(bbox_to_anchor=(1.3, 2.95))
plt.show()

# fig 4.3
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.subplot(221)
plt.ylim(0.65, 0.85)
plt.title("Accuracy")

name_list = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
num_list = [0.799292, 0.742121, 0.677272, 0.760399]
x = list(range(len(num_list)))
width = 0.3

plt.bar(x, num_list, width=width, fc='blue', tick_label=name_list)

plt.subplot(222)
plt.ylim(0.5, 0.7)
plt.title("Precision")

name_list = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
num_list = [0.659341, 0.597015, 0.504545, 0.62963]
x = list(range(len(num_list)))
width = 0.3

plt.bar(x, num_list, width=width, fc='blue', tick_label=name_list)

plt.subplot(223)
plt.ylim(0.6, 0.8)
plt.title("Recall")

name_list = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
num_list = [0.764331, 0.722892, 0.689441, 0.712575]
x = list(range(len(num_list)))
width = 0.3

plt.bar(x, num_list, width=width, fc='blue', tick_label=name_list)

plt.subplot(224)
plt.ylim(0.5, 0.8)
plt.title("F1-score")

name_list = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
num_list = [0.707965, 0.653951, 0.582677, 0.668539]
x = list(range(len(num_list)))
width = 0.3

plt.bar(x, num_list, width=width, fc='blue', tick_label=name_list)


plt.legend(bbox_to_anchor=(1.3, 2.9))
plt.show()

# fig 4.4
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.subplot(221)
plt.ylim(0.7, 0.90)
plt.title("Accuracy")

name_list = ['With Content', 'Without Content']
num_list = [0.847530, 0.810382]
x = list(range(len(num_list)))
plt.bar(x, num_list, width=0.2, fc='blue', tick_label=name_list)

plt.subplot(222)
plt.ylim(0.6, 0.8)
plt.title("Precision")

name_list = ['With Content', 'Without Content']
num_list = [0.764368, 0.702703]
x = list(range(len(num_list)))
plt.bar(x, num_list, width=0.2, fc='blue', tick_label=name_list)

plt.subplot(223)
plt.ylim(0.7, 0.85)
plt.title("Recall")

name_list = ['With Content', 'Without Content']
num_list = [0.796407, 0.77381]
x = list(range(len(num_list)))
plt.bar(x, num_list, width=0.2, fc='blue', tick_label=name_list)

plt.subplot(224)
plt.ylim(0.6, 0.85)
plt.title("F1-score")

name_list = ['With Content', 'Without Content']
num_list = [0.780059, 0.736544]
x = list(range(len(num_list)))
plt.bar(x, num_list, width=0.2, fc='blue', tick_label=name_list)
plt.show()

# fig 4.5
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)
plt.subplot(221)
plt.ylim(0.80, 0.90)
plt.title("Accuracy")

name_list = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
num_list = [0.846530, 0.855918, 0.842448, 0.835510]
x = list(range(len(num_list)))
width = 0.3

plt.bar(x, num_list, width=width, fc='blue', tick_label=name_list)

plt.subplot(222)
plt.ylim(0.7, 0.9)
plt.title("Precision")

name_list = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
num_list = [0.791908, 0.8375, 0.730769, 0.735484]
x = list(range(len(num_list)))
width = 0.3

plt.bar(x, num_list, width=width, fc='blue', tick_label=name_list)

plt.subplot(223)
plt.ylim(0.6, 0.9)
plt.title("Recall")

name_list = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
num_list = [0.774011, 0.748603, 0.76, 0.74026]
x = list(range(len(num_list)))
width = 0.3

plt.bar(x, num_list, width=width, fc='blue', tick_label=name_list)

plt.subplot(224)
plt.ylim(0.7, 0.9)
plt.title("F1-score")

name_list = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
num_list = [0.782857, 0.79056, 0.745098, 0.737864]
x = list(range(len(num_list)))
width = 0.3

plt.bar(x, num_list, width=width, fc='blue', tick_label=name_list)


plt.legend(bbox_to_anchor=(1.3, 2.9))
plt.show()

# fig 4.6
x_data = [str(x) for x in range(1970, 2022)]
y_raw = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
             [], [], [4441643], [642933], [], [4578146, 5172904], [], [], [4859604], [], [], [], [643126], [], [], [],
             [], [], [], [], [], [], [], [], []]

y_data = [-len(x) for x in y_raw]
y2_raw = [[], [1260924, 2458352, 2930448], [804549, 3149019, 4628284], [1078887, 3530003],
             [2408827, 2466566, 3149927], [917452, 1260890, 2973579], [803326, 1017380, 1025050, 3147759],
             [281248, 305217, 3529498, 3740327, 3740343], [1023032, 1178053, 2410514, 3148843], [2410953, 3740379],
             [1018347, 1024694, 4629032], [282881, 2411995], [2363762, 3094534, 4123994, 4124191],
             [2406845, 3740385, 4796402], [278572, 3093781, 4123577, 4628473, 4796767], [], [917587, 3148727, 4629192],
             [1029710, 1176952, 1314217, 3149751], [2363821, 2411672, 3354820, 4124050, 4183167, 4797095],
             [4124291, 4797073], [905299, 3776522, 4123334], [643116, 643240, 4124087, 4124302],
             [593229, 1021519, 2482865, 4123944, 4124262], [2409083, 4110743], [444200, 1783519, 4123999],
             [1190818, 2547769], [1189440, 1190891, 2363133, 2708907, 4796197], [], [642793, 1190290, 1998152, 4053341],
             [3928013], [642933, 3725279], [277539, 3937735, 4023204, 4023205, 4948130], [], [4182930], [], [],
             [386501, 2280949], [3530780], [2370378], [], [2443439], [1189835], [1989292], [3220275], [2409475], [], [],
             [], [], [], [], []]
y_data2 =  [len(x) for x in y2_raw]

y3_raw = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
             [], [], [], [642933], [], [], [], [], [], [], [], [], [], [], [], [],
             [], [], [], [], [], [], [], [], []]

y_data3 = [-len(x) for x in y3_raw]

fig1, ax = plt.subplots()
plt.title("Number of paper")
ax.plot(x_data, y_data, label='Advisee')
ax.plot(y_data2, label='Advisor')
ax.plot(y_data3, label='Shared paper')
xticks=list(range(0, len(x_data), 10))
xlabels=[x_data[x] for x in xticks]
xticks.append(len(x_data))
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=10)

plt.legend()
plt.show()