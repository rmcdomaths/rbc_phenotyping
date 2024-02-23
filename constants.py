import numpy as np

class DataFile(object):

    def __init__(self, file):
        self.filename = file
        self.name = file.name

    def get_class(self):
        # name = self.name
        # seen_underscore = False
        # for n, c in enumerate(name):
        #     if not seen_underscore and c == '_':
        #         seen_underscore = True
        #         continue
        #     if seen_underscore and c.isnumeric():
        #         break
        # name_class = name[:n]
        name_class = self.filename.parent.name

        return name_class

    def get_name(self):
        name = self.name
        return name[:-4]

    def get_sde(self):
        c = self.get_class()
        if not (c[0].isnumeric() or c == '-'):
            sde = np.nan
        else:
            sde = []
            for c in self.name:
                if c == '-' or c.isnumeric() or c == '.':
                    sde.append(c)
                else:
                    break
            sde = float(''.join(sde))
        return sde

    def get_number(self):
        number = []
        for c in self.get_name()[::-1]:

            if c.isnumeric():
                number.append(c)
            else:
                break
        return int(round(float(''.join(number[::-1]))))
