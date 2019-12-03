现在理论上只需要一个att\_feat就可以了，shape是batch * 1 * (2048 + 2048 + 5 + 2048 + 25)
不过最好能再传一个att\_mask, batch * 1的全1 tensor
