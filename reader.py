import os
from matplotlib import pyplot as plt

class Encounter:
    def __init__(self, sma, ltan, ecc, aop, datalines):
        self.sma = sma
        self.ltan = ltan
        self.ecc = ecc
        self.aop = aop
        self.data = datalines

    @property
    def ltdn(self):
        return (self.ltan + 12) % 24

    def __repr__(self):
        return f'''
        SMA = {self.sma}
        LTDN = {self.ltdn}
        ECC = {self.ecc}
        AOP = {self.aop}
        '''


def orb_parser(name: str) -> "tuple[float, float, float, float]":
    buffer = []

    values = []

    c: str
    for c in (name + "0x"):
        if c.isnumeric():
            buffer.append(c)

        elif len(buffer) > 0:
            b = "".join(buffer)
            if b[0] == "0":
                values.append(float(b[0] + "." + b[1:]))
            else:
                values.append(float(b[:-1] + "." + b[-1:]))
            buffer.clear()

    return tuple(values)


encounters = []

for (dirpath, dirnames, filenames) in os.walk("."):
    csv = [i for i in filenames if i[-3:] == "csv"]
    if not csv:
        continue
    with open(f"{dirpath}/{csv[0]}") as f:
        encounters.append(Encounter(*orb_parser(dirpath[2:]), f.readlines()))

encounters10_0 = [e for e in encounters if e.sma == 6878 and e.ecc == 0]

smas = [e.ltdn for e in encounters10_0]
times_img = [float(e.data[2].split(",")[1]) for e in encounters10_0]
times_dl = [float(e.data[2].split(",")[3]) for e in encounters10_0]

plt.plot(smas, times_img, label="Imaging")
plt.plot(smas, times_dl, label="DL")
plt.legend()
plt.ylabel("Number of events per day")
plt.xlabel("Semi-Major Axis (km)")
plt.show()
