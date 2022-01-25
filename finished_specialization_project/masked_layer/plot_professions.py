import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display
import plotly.figure_factory as ff
import pandas as pd

df_mbert = pd.read_csv('data/mbert1_professions.csv')
df_nbbert = pd.read_csv('data/nbbert1_professions.csv')
df_norbert = pd.read_csv('data/norbert1_professions.csv')

#HAN
df_mbert_han = df_mbert.round(3).sort_values(by=['Differanse'],ascending=False).head(5)
df_nbbert_han = df_nbbert.round(3).sort_values(by=['Differanse'],ascending=False).head(5)
df_norbert_han = df_norbert.round(3).sort_values(by=['Differanse'],ascending=False).head(5)

#HUN
df_mbert_hun = df_mbert.round(3).sort_values(by=['Differanse'],ascending=True).head(5)
df_nbbert_hun = df_nbbert.round(3).sort_values(by=['Differanse'],ascending=True).head(5)
df_norbert_hun = df_norbert.round(3).sort_values(by=['Differanse'],ascending=True).head(5)

#SAMLET

# Top 50 hun 
df_mbert_results = df_mbert_han.append(df_mbert_hun)
df_nbbert_results = df_nbbert_han.append(df_nbbert_hun)
df_norbert_results = df_norbert_han.append(df_norbert_hun)

print("mbert")
display(df_mbert_results)
print("nb bert under her")
display(df_nbbert_results)
print("norbert under her")
display(df_norbert_results)


"""
def render_mpl_table(data, col_width=6.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax

fig,ax = render_mpl_table(df_mbert_results, header_columns=0, col_width=2.0)
fig.savefig("results_mbert.png")"""