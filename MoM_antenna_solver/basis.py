def basis_value(node, seg, x):
    if seg == node - 1:
        return x
    if seg == node:
        return 1.0 - x
    return 0.0


def basis_derivative(node, seg, dl):
    if seg == node - 1:
        return 1.0 / dl[seg]
    if seg == node:
        return -1.0 / dl[seg]
    return 0.0
