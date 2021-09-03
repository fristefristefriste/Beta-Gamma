import numpy as np
from numpy import linalg as la
import re
from typing import *
import doctest


"""
Limited to real or vector-valued transformations. :[
"""

DOMAIN = 0
CODOMAIN = 1
SIG = 2
EXPR = 3


class BadInputException(Exception):
    pass


def parse_func(s: str) -> (int, int, List[str], List[str]):
    """
    f:R^4->R^2; f(x,y,z,w)=(x-y,x+z)
    """
    lst = s.split('; ')
    lst[0] = lst[0][2:]
    f = lst[1].split('=')
    domain = int(lst[0][2])
    codomain = int(lst[0][7])  # garbage
    sig = f[0][2:-1].split(',')
    expr = f[1][1:-1].split(',')
    return domain, codomain, sig, expr


def parse_basis(s: str) -> List[List[Any]]:
    """
    >>> parse_basis('[1,0],[0,1]')
    [[1, 0], [0, 1]]
    """
    regex = r'\[(!\[).*\]'
    searches = re.search(regex, s)
    basis = []
    for vec in searches.groups():
        num_strings = vec[1:-1].split(',')
        v = []
        for num in num_strings:
            v.append(int(num))
        basis.append(v)
    return basis


def convert_to_python(s: str) -> list:
    """
    converts s to math expression in python syntax
    ASSUMES ALL ALPHABETICAL CHARACTERS ARE VARIABLES.
    ONLY CONVERTS IF LEGIT MATH EXPRESSION
    >>> convert_to_python('7x+8z')
    ['7', '*', 'x', '+', '8', '*', 'z']
    >>> convert_to_python('xy')
    ['x', '*', 'y']
    """
    new = []
    for i, char in enumerate(s[:-1]):
        new.append(char)
        if char.isalnum() and s[i+1].isalnum():
            new.append('*')
    new.append(s[-1])
    return new


def evaluate(ex: list, vector: list, sig: list) -> Any:
    """
    [x,-,y],[1,0,0],(x,y,z) -> def f(x,y): return x-y -> return f(input)
    >>> evaluate(['x', '-', 'y'], [0, 1], ['x', 'y'])
    -1
    """
    ex1 = ex[:]
    d = dict(zip(sig, vector))
    for i, char in enumerate(ex):
        if char in sig:
            ex1[i] = str(d[char])
    expr = ''.join(ex1)
    if not any(char.isdigit() for char in expr):
        raise BadInputException

    # scaling
    # addition
    # subtraction
    return eval(expr)  # A little bit questionable


def verify_linear(t: str) -> bool:
    regex = r'0|([0-9]*[a-z][0-9]*)((-|\+)[0-9]*[a-z][0-9]*)*'
    match = re.search(regex, t)
    if not match:
        return False
    a, b = match.span()
    return b - a == len(t)


def verify_basis(b):
    """Checks that set is a basis."""

    pass


def convert_to_matrix(sig: List[str], t: List[str], beta, gamma):
    """ args: x-y, x+z' -> [T]_b^b"""
    # entries = t[1:-1].split(', ')
    # beta = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    coords = [convert_to_python(exp) for exp in t]

    t_betas = []
    for b in beta:
        t_beta = []
        for coord in coords:
            t_beta.append(evaluate(coord, b, sig))
        t_betas.append(t_beta)
    final = []
    gamma_temp = np.array(gamma)
    np.transpose(gamma_temp)
    for t_b in t_betas:
        nt_b = np.array(t_b)
        final.append(la.solve(gamma_temp, nt_b.transpose()))
    print(np.array(final).transpose())


def generate_standard(n) -> List[Any]:
    beta = []
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        beta.append(e)
    return beta


if __name__ == '__main__':
    """
    f:R^2->R^2; f(x,y)=(x-y,x+y)
    """
    t = input("Enter a linear transformation.\n")
    # t = 'f:R^3->R^3; f(x,y,z)=(3x-y,z-x,z-y)'
    t1 = parse_func(t)
    # need to flatten list input

    for expr in t1[EXPR]:
        if not verify_linear(expr):
            raise BadInputException
    b1 = input("Enter a basis of the domain.\n")
    b2 = input("Enter a basis of the codomain.\n")
    if b1.lower() == 'standard':
        b1 = generate_standard(t1[DOMAIN])
    if b2.lower() == 'standard':
        b2 = generate_standard(t1[CODOMAIN])
    # convert_to_matrix(t1[SIG], t1[EXPR], [[1,0,0],[0,1,0],[0,0,1]], [[1,0,0],[0,1,0],[0,0,1]])
    convert_to_matrix(t1[SIG], t1[EXPR], b1, b2)

