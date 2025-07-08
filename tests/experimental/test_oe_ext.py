import torch
import opt_einsum as oe
# import oe_ext

def f(a,b,c):
    return oe.contract('ab,bc,ca',a,b,c)

a,b,c= torch.rand(3,3), torch.rand(3,3), torch.rand(3,3)

f_comp= torch.compile(f)
print(f_comp(a,b,c))


eq = "ij,jk,kl,lm,mn->ni"
#           A       B       C       D       E
shapes = [(9, 5), (5, 5), (5, 5), (5, 5), (5, 8)]
# mark the middle three arrays as constant
constants = [1, 2, 3]
# generate the constant arrays
B, C, D = [torch.rand(*shapes[i]) for i in constants]
# supplied ops are now mix of shapes and arrays
ops = (9, 5), B, C, D, (5, 8)

def expr_f(a,b,c,d):
    expr = oe.contract_expression(eq, *ops, constants=constants)
    return expr(a, b) + expr(c, d)

A1, E1 = torch.rand(*shapes[0]), torch.rand(*shapes[-1])

expr_f_comp= torch.compile(expr_f)
print(expr_f(A1,E1,A1*2,E1*2))
