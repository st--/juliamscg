import numpy as np
import itertools
from collections import OrderedDict

class InvalidFrame(Exception): pass

def _parsebool(s):
    if s.lower() in ['t', 'true']: return True
    elif s.lower() in ['f', 'false']: return False
    else: raise ValueError("invalid literal for bool")

_convert = {'R': float, 'I': int, 'L': _parsebool, 'S': str}

class XYZData:
    def __init__(self):
        self.issingle = None
        self.fid = None # total num of frames if not single, else frame num
        self.extended = None
        self.lattice = None
        self.param = {}
        self.props = None
        self.data = OrderedDict()
        self.next3 = None

    def has_column(self, key):
        return key in self.data
    def has_param(self, key):
        return key in self.param

    def __getattr__(self, key):
        if key in self.data and key in self.param:
            raise AttributeError("key %r both in data and param" % key)
        elif key in self.data:
            return self.data[key]
        elif key in self.param:
            return self.param[key]
        else:
            raise AttributeError("%s object has no param or data column %r" %
                    (self.__class__.__name__, key))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return tuple(self.data[k] for k in key)
        else:
            return self.data[key]

class XYZ(XYZData):
    def __init__(self, iter, extended=True, next3=None):
        XYZData.__init__(self)
        self.fid = 0 # number of the current frame; eventually number of frames
        self.extended = extended # extended XYZ format?
        self.lattice = [] # list of 3x3 arrays
        self.param = {} # dictionary of lists of values per frame
        self.props = None # dictionary of (type, # of columns) tuple
        self.data = OrderedDict() # dictionary of lists of arrays per frame
        self.next3 = next3 # name for three more columns in non-extended mode
        self.raw = [] # raw lines
        self.load(iter)

    def __len__(self):
        return self.fid

    def assingle(self, frame=0):
        if not -self.fid <= frame < self.fid:
            raise ValueError("frame number out of range")

        if frame < 0:
            frame = self.fid + frame

        single = XYZData()
        single.issingle = True
        single.fid = frame
        single.extended = self.extended
        single.lattice = self.lattice[frame]
        for key in self.param:
            single.param[key] = self.param[key][frame]
        single.props = self.props
        for key in self.data:
            single.data[key] = self.data[key][frame]
        single.next3 = self.next3
        return single

    def load(self, iter):
        self.issingle = False
        try:
            while True:
                line = iter.next()
                self.raw.append([line])
                try:
                    n = int(line)
                except ValueError:
                    raise InvalidFrame("number of atoms not given")
                self.loadframe(iter, n)
                self.fid += 1
        except StopIteration:
            pass

        for key in self.data:
            self.data[key] = np.array(self.data[key])

    def parsecomment(self, header):
        found = set()

        while header:
            if ' ' in header:
                param, header = header.split(' ', 1)
            else:
                param, header = header, ""

            if '"' in param: # deal with quoting of strings containing spaces
                while True:
                    idx = header.find('"')
                    if idx == -1:
                        raise ValueError("unbalanced quote character")
                    if header[idx-1] == '\\': # escaped quote
                        continue
                    else:
                        break
                param += ' ' + header[:idx+1] # split() had removed the space
                header = header[idx+1:]

            if '=' in param: # found another parameter
                key, val = map(str.strip, param.split('='))
                if val and val[0] == val[-1] == '"':
                    val = val[1:-1] # remove quotes

                if key in found:
                    raise ValueError("key not unique: %s" % key)

                if key == 'Lattice':
                    val = np.array(val.split(), float).reshape((3, 3))
                    self.lattice.append(val.T) # was in Fortran order!

                elif key == 'Properties':
                    self.parseprops(val)

                else:
                    try: val = int(val)
                    except ValueError:
                        try: val = float(val)
                        except ValueError:
                            pass
                    if key not in self.param:
                        # new parameter introduced
                        # fill in None value for previous frames
                        self.param[key] = [None] * self.fid
                    self.param[key].append(val)

                found.add(key)

        if 'Lattice' not in found or 'Properties' not in found:
            if self.fid == 0:
                # plain XYZ format
                self.extended = False
            else:
                raise InvalidFrame("extended XYZ needs Lattice and Properties")

        for key in set(self.param) - found:
            # parameter suddenly no longer present
            # fill in None value instead
            self.param[key].append(None)

    def parseprops(self, properties):
        fields = properties.split(':')
        if len(fields) % 3 != 0:
            raise ValueError("Properties not in name:type:col:... format")
        props = OrderedDict()
        for name, ptype, cols in zip(
                fields[::3], fields[1::3], map(int, fields[2::3])):
            if ptype not in 'RILS':
                raise ValueError("Property type must be one of R, I, L, S")
            props[name] = (_convert[ptype], cols)
        if self.props is None:
            self.props = props
            for key in self.props:
                self.data[key] = []
        elif self.props != props:
            raise ValueError("inconsistent Properties")

    def loadframe(self, iter, n):
        _raw_frame = self.raw[self.fid]
        header = iter.next()
        _raw_frame.append(header)
        if self.extended:
            self.parsecomment(header)

        self_data = self.data
        _array = np.array

        if self.extended: # might have changed
            data = {key: [] for key in self.props}
            props = self.props.items()
            for i in range(n):
                line = iter.next()
                _raw_frame.append(line)
                fields = line.split()
                j = 0
                for name, (converter, col) in props:
                    val = map(converter, fields[j:j+col])
                    if col == 1:
                        val = val[0]
                    data[name].append(val)
                    j += col
            for key in self_data:
                self_data[key].append(_array(data[key]))

        else: # plain mode
            next3 = self.next3
            if 'species' not in self_data: self_data['species'] = []
            if 'pos' not in self_data: self_data['pos'] = []
            if next3:
                if next3 not in self_data: self_data[next3] = []
                species, pos, oth = zip(*((fields[0], fields[1:4], fields[4:7])
                    for fields in map(str.split, itertools.islice(iter, n))))
            else:
                species, pos = zip(*((fields[0], fields[1:4])
                    for fields in map(str.split, itertools.islice(iter, n))))

            #species = []
            #pos = []
            #oth = []
            #for i in range(n):
            #    line = iter.next()
            #    fields = line.split()
            #    species.append(fields[0])
            #    pos.append(fields[1:4])
            self_data['species'].append(np.array(species))
            self_data['pos'].append(np.array(pos, float))
            if next3:
                self_data[next3].append(np.array(oth, float))

def load_frame(f):
    nstr = f.readline()
    try:
        n = int(nstr)
    except ValueError:
        raise InvalidFrame
    comment = f.readline()
    data = []
    for i in range(n):
        line = f.readline()
        data.append(map(float, line.split()[1:7]))
    adata = np.array(data)
    pos, force = adata[:,:3], adata[:,3:]
    return pos, force

def load_all_frames(f):
    posdata = []
    fordata = []
    try:
        while True:
            pos, force = load_frame(f)
            posdata.append(pos)
            fordata.append(force)
    except InvalidFrame:
        pass
    return np.array(posdata), np.array(fordata)

def load(xyz, **kwargs):
    if isinstance(xyz, str):
        if '\n' in xyz and xyz[:xyz.index('\n')].isdigit():
            xyz = iter(xyz.split('\n'))
        else:
            xyz = open(xyz, 'r')
    return XYZ(xyz, **kwargs)

FRAMESIZE, HEADER, ATOMLINE = range(3)

def readxyzpos(f):
    nextlinetype = FRAMESIZE
    framesize = None
    headers = []
    allpos = []
    frame = []
    i = 0
    while True:
        line = f.readline()
        if not line: break
        if nextlinetype == FRAMESIZE:
            i += 1
            #if i % 1000 == 0:
            #    print i
            if frame:
                allpos.append(frame)
                frame = []
            framesize = int(line.strip())
            nextlinetype = HEADER
        elif nextlinetype == HEADER:
            headers.append(line.strip())
            nextlinetype = ATOMLINE
        elif nextlinetype == ATOMLINE:
            fields = line.split()
            frame.append(map(float, fields[1:4]))
            if len(frame) == framesize:
                nextlinetype = FRAMESIZE
    if frame:
        allpos.append(frame)
    return np.array(allpos), headers

def fastloadpos(xyz, returnheaders=False):
    if isinstance(xyz, str):
        if '\n' in xyz and xyz[:xyz.index('\n')].isdigit():
            xyz = iter(xyz.split('\n'))
        else:
            xyz = open(xyz, 'r')
    if returnheaders:
        return readxyzpos(xyz)
    else:
        return readxyzpos(xyz)[0]

def fastloadheader(xyz):
    if isinstance(xyz, str):
        if '\n' in xyz and xyz[:xyz.index('\n')].isdigit():
            xyz = iter(xyz.split('\n'))
        else:
            xyz = open(xyz, 'r')

    allheaders = []
    while True:
        numberline = xyz.readline()
        if not numberline: break

        framesize = int(numberline.strip())
        header = xyz.readline().strip()
        allheaders.append(header)

        for i in range(framesize): xyz.readline() # skip atoms

    return allheaders

def getfield(headers, field, parser=lambda x:x):
    values = []
    fieldeq = field + '='
    labellen = len(fieldeq)
    for h in headers:
        val = [parser(f[labellen:]) for f in h.split() if f.startswith(fieldeq)]
        values.append(val)
    return values

