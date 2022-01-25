import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display

df = pd.read_csv('norbert_100_adjectives.csv')

df= df.round(decimals=3)

    
df.to_csv(r'rounded_adjectives.csv')

"""
plt.scatter(new_df['Differanse'],new_df['Adjektiv'], )
plt.rcParams["figure.figsize"] = (80,80)

for x,y in zip(new_df['Differanse'],new_df['Adjektiv']):
    label = f"({round(x,5)},{y})"
    plt.annotate(label,
                 (x,y),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center')

plt.show()
"""