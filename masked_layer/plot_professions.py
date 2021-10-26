import pandas as pd
from IPython.display import display

df = pd.read_pickle('/home/login/Fordypningsoppgave/masked_layer/gendered_professions2.pkl')
display(df)

# Top 100 han
df_han = df.sort_values(by=['Differanse'],ascending=False).head(100)
# Top 50 hun 
df_hun = df.sort_values(by=['Differanse'],ascending=True).head(100)

display(df_han)
display(df_hun)
