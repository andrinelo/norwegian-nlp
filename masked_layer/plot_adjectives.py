import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display

df = pd.read_pickle('/home/login/Fordypningsoppgave/masked_layer/gendered_adjectives.pkl')
display(df)

# Top 100 han
df_han = df.sort_values(by=['Differanse'],ascending=False).head(50)
# Top 50 hun 
df_hun = df.sort_values(by=['Differanse'],ascending=True).head(50)

new_df = df_han.append(df_hun)

display(df_han)
display(df_hun)
    
new_df.to_csv(r'adjectives.csv')

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