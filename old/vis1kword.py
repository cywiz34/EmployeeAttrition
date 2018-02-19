import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import decomposition


df = pd.DataFrame.from_csv('data_1k_word.csv', index_col=None)


corr = df.corr()
corr = (corr)
corr.to_csv("1k_word_corr.csv")
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
plt.show()