# --- Core object definitions ---


class Linker:
    def __init__(self, label, atoms, properties=None):
        self.label = label
        self.atoms = atoms
        self.ccoords = None
        self.fcoords = None
        self.bond_list = None
        self.center = None
        #properties is a dictionary
        #can include charge, spin, etc properties from QM calculations
        #can include force field parameters
        #metal ions
        #can include other info
        self.properties = properties or {} 

class LinkerFragment:
    def __init__(self, label, atoms, properties=None):
        self.label = label
        self.atoms = atoms
        self.ccoords = None
        self.fcoords = None
        self.bond_list = None
        self.center = None
        #properties is a dictionary
        #can include charge, spin, etc properties from QM calculations
        #can include force field parameters
        #can include other info
        self.properties = properties or {}

class LinkerCenter:
    def __init__(self, label, atoms, properties=None):
        self.label = label
        self.atoms = atoms
        self.ccoords = None
        self.fcoords = None
        self.bond_list = None
        self.center = None
        #properties is a dictionary
        #can include charge, spin, etc properties from QM calculations
        #can include force field parameters
        #can include other info
        self.properties = properties or {}


class Termination:
    def __init__(self, label, atoms, properties=None):
        self.label = label
        self.atoms = atoms
        self.ccoords = None
        self.fcoords = None
        self.bond_list = None
        self.center = None
        #properties is a dictionary
        #can include charge, spin, etc properties from QM calculations
        #can include force field parameters
        #can include other info
        self.properties = properties or {}

class Net:
    def __init__(self, name=None):
        self.name = name
        self.vertices = None
        self.edges = None
        self.faces = None
        self.pores = None
        self.net_graph = None

class Framework:
    def __init__(self, net=None, nodes=None, linkers=None, terminations=None):
        self.spacegroup = None
        self.lattice_parameters = None
        self.lattice_vectors = None
        self.net = net
        self.primitive_cell = None
        self.supercell = None
        self.nodes = nodes or []
        self.linkers = linkers or []
        self.terminations = terminations or []



        

