import IPython

def corr_style(df):
    out = df.style.background_gradient(cmap='coolwarm').format(precision =2)
    IPython.display.display(out)

def column_to_pos(df,columns, pos=0):
    assert isinstance(columns,list)
    for col in columns[::-1]:
        popped = df.pop(col)
        df.insert(pos, col ,popped)
    
    