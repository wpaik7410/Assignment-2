from sklearn import datasets
iris = datasets.load_iris()

import pandas as pd
data = { "weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26], "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10}
PlantGrowth = pd.DataFrame(data)


def find_target_name(n, target_names) :
    return target_names[n]
new_iris=pd.DataFrame(iris['data'])
new_iris.columns=iris['feature_names']
new_iris['Species']=find_target_name(iris['target'], iris['target_names'])
iris=new_iris

import matplotlib.pyplot as plt


# 1.Using the iris dataset...
#  a. Make a histogram of the variable Sepal.Width.

plt.hist(iris['sepal width (cm)'], bins=10, edgecolor='black')
plt.show()


#  b.Based on the histogram from #1a, which would you expect to be higher, the mean or the median? Why?
#  -> mean will be lower because left and right side from the middle are not balanced. It is leaning to left side slightly.


#  c.Confirm your answer to #1b by actually finding these values.

print(iris['sepal width (cm)'].mean())
print(iris['sepal width (cm)'].median())


#  d.Only 27% of the flowers have a Sepal.Width higher than ________ cm.

print("%scm"%iris['sepal width (cm)'].quantile([0.73]).values[0])


#  e.Make scatterplots of each pair of the numerical variables in iris (There should be 6 pairs/plots).

fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10,10))
axes = ax.flatten()
num=0

for x in iris.columns :
    for y in iris.columns[1:] :
        if x!=y and x!="Species" and y!="Species" :
            axes[num].scatter(iris[x], iris[y])
            axes[num].set_xlabel(x)
            axes[num].set_ylabel(y)
            axes[num].set_title("%s vs %s"%(x,y))
            num+=1

plt.tight_layout()
plt.show()


#  f.Based on #1e, which two variables appear to have the strongest relationship? And which two appear to have the weakest relationship?
#  -> strongest relationship : sepal length (cm) vs petal length (cm)
#  -> weakest relationship : sepal length (cm) vs sepal width (cm)



# 2. Using the PlantGrowth dataset...
#  a. Make a histogram of the variable weight with breakpoints (bin edges) at every 0.3 units, starting at 3.3.

import numpy as np
bins_list = np.arange(3.3, float(PlantGrowth['weight'].max())+0.3, 0.3)
plt.figure(figsize=(7,7))
plt.hist(PlantGrowth["weight"], bins=bins_list, edgecolor='black')
plt.xlabel("Weight")
plt.ylabel("Freq.")
plt.title("Weight vs Freq.")
plt.show()


#  b. Make boxplots of weight separated by group in a single graph.

plt.figure(figsize=(5,5))
grouped_data = [PlantGrowth[PlantGrowth['group']=="ctrl"]['weight'],PlantGrowth[PlantGrowth['group']=="trt1"]['weight'],PlantGrowth[PlantGrowth['group']=="trt2"]['weight']]
plt.boxplot(grouped_data, labels=['ctrl','trt1','trt2'])

plt.xlabel("Group")
plt.ylabel("Weight")
plt.title("Group vs Weight")
plt.show()


#  c. Based on the boxplots in #2b, approximately what percentage of the "trt1" weights are below the minimum "trt2" weight?
#  -> except outlier of "trt1", max of "trt1" is below min of "trt2". Therefore, more than 90% of "trt1" is below min "trt2".


#  d. Find the exact percentage of the "trt1" weights that are below the minimum "trt2" weight.

trt2_min = PlantGrowth[PlantGrowth['group']=="trt2"]['weight'].min()
trt1_cnt = PlantGrowth[(PlantGrowth['group']=="trt1") & (PlantGrowth['weight']<trt2_min)]['weight'].count()
print(trt1_cnt*100/PlantGrowth[PlantGrowth['group']=="trt1"]['weight'].count())


#  e. Only including plants with a weight above 5.5, make a barplot of the variable group. 
#     Make the barplot colorful using some color palette (in R, try running ?heat.colors and/or check out https://www.r-bloggers.com/palettes-in-r/).

from matplotlib import cm
import random

colors = plt.get_cmap('viridis')
new_data = PlantGrowth[PlantGrowth['weight']>5.5]
plt.figure(figsize=(5,5))
x=[0,1,2]
x_label=['ctrl', 'trt1', 'trt2']
y=[new_data[new_data['group']=="ctrl"]['weight'].count(),new_data[new_data['group']=="trt1"]['weight'].count(),new_data[new_data['group']=="trt2"]['weight'].count()]
plt.bar(x,y,color=[color(i*100) for i in x])
#, color=colors, cmap=
plt.xticks(x,x_label)
plt.xlabel("Group")
plt.ylabel("Weight")
plt.title("Group vs Weight")
plt.show()