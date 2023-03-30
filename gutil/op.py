import IPython

def corr_style(df):
    out = df.style.background_gradient(cmap='coolwarm').format(precision =2)
    IPython.display.display(out)