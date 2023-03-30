import sympy


def opt_x(expr, var):
    '''
    just solve derivative =0 withouth cechking
    '''
    deriv = sympy.diff(expr, var)
    sol = sympy.solve(deriv, var)
    assert len(sol)==1, len(sol)
    return sol[0]

def opt_y(expr,var):
    '''
    just substitute value from above
    '''
    x = opt_x(expr,var)
    return expr.subs(var, x).simplify()