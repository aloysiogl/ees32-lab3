def octal2poly(octal, m):
    poly = []
    for num in octal:
        bi = dec2bi(num, 3)
        for bit in bi:
            poly.append(bit)

    while len(poly) > m + 1:
        poly.pop(0)

    poly = poly[::-1]
    return poly


def dec2bi(dec, sz):
    bi = []
    while True:
        bi.append(dec % 2)
        dec = dec // 2
        if dec == 0:
            break

    while len(bi) < sz:
        bi.append(0)

    bi = bi[::-1]
    return bi


if __name__ == '__main__':
    m = 3
    g1 = [1, 3]
    g2 = [1, 5]
    g3 = [1, 7]

    print("G1 = ", octal2poly(g1, m))
    print("G2 = ", octal2poly(g2, m))
    print("G2 = ", octal2poly(g3, m))
