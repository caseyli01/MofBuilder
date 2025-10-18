import numpy as np
import networkx as nx
from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from veloxchem.errorhandler import assert_msg_critical
import mpi4py.MPI as MPI
import sys
import time
from .other import safe_copy
from ..utils.environment import get_data_path
from ..utils.fetch import fetch_pdbfile
from pathlib import Path
from .linker import FrameLinker
from .write import MofWriter
from ..io.pdb_reader import PdbReader
from ..md.linkerforcefield import LinkerForceFieldGenerator
from ..md.gmxfilemerge import GromacsForcefieldMerger
from ..md.solvationbuilder import SolvationBuilder
from ..visualization.viewer import Viewer
from ..md.setup import OpenmmSetup
from .defects import TerminationDefectGenerator

class Framework:
    def __init__(self, comm=None, ostream=None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream or OutputStream(sys.stdout if self.rank == mpi_master() else None)

        self.mofwriter = MofWriter(comm=self.comm, ostream=self.ostream)

        #MD preparation
        self.framework_data = None  #merged data for the whole framework, generated in write() 
        self.framework_fcoords_data = None  #merged fractional coordinates for the whole framework, generated in write()

        self.solvationbuilder = SolvationBuilder(comm=self.comm, ostream=self.ostream)
        self.solvents = []  #list of solvent names or xyz files
        self.solvents_molecules = []  #list of solvent molecules
        self.solvents_proportions = []  #list of solvent proportions
        self.solvents_quantities = []  #list of solvent quantities
        #will be set
        self.solvated_gro_file = None
        #MD simulation

        #others for output and saving
        self.target_directory = None  #will be set to default path if None
        self.data_path = None  #will be set to default path if None

        self.save_files = False
        self.linker_ff_name = "Linker"
        self.linker_charge = None
        self.linker_multiplicity = None
        self.linker_reconnect_drv = 'xtb'
        self.linker_reconnect_opt = True
        self.provided_linker_itpfile = None  #if provided, will map directly

        #debug
        self._debug = False

        #will be set by MofBuilder.build()
        self.mof_family = None
        self.node_metal = None
        self.dummy_atom_node = None
        self.net_spacegroup = None
        self.net_cell_info = None
        self.net_unit_cell = None
        self.node_connectivity = None
        self.linker_connectivity = None
        self.linker_fragment_length = None
        self.node_data = None
        self.dummy_atom_node_dict = None
        self.termination_data = None
        self.termination_X_data = None
        self.termination_Y_data = None
        self.frame_unit_cell = None
        self.frame_cell_info = None
        self.graph = None
        self.cleaved_graph = None
        self.graph_index_name_dict = None
        self.graph_matched_vnode_xind = None
        self.supercell_info = None
        self.supercell = None
        self.termination = None
        self.termination_name = None
        self.add_virtual_edge = None
        self.virtual_edge_max_neighbor = None
        self.sc_unit_cell = None
        self.sc_unit_cell_inv = None

        self.clean_unsaturated_linkers = False
        self.update_node_termination = True
        self.unsaturated_linkers = []
        self.unsaturated_nodes = []
        self.matched_vnode_xind = None
        self.xoo_dict = None

        self.residues_info = None  #dictionary of residue name and quantity
        self.solvents_dict = None  #dictionary of solvents info after solvation



    def exchange(self, exchange_indices=[],exchange_node_pdbfile=None, exchange_linker_pdbfile=None, exchange_linker_molecule=None):
        self.defectgenerator = TerminationDefectGenerator(comm=self.comm,ostream=self.ostream)
        self.defectgenerator.use_termination = self.termination
        self.defectgenerator.linker_connectivity = self.linker_connectivity
        self.defectgenerator.node_connectivity = self.node_connectivity + self.virtual_edge_max_neighbor if self.add_virtual_edge else self.node_connectivity
        self.defectgenerator._debug = self._debug
        self.defectgenerator.eG_index_name_dict = self.graph_index_name_dict
        self.defectgenerator.eG_matched_vnode_xind = self.graph_matched_vnode_xind
        self.defectgenerator.sc_unit_cell_inv = self.sc_unit_cell_inv
        self.defectgenerator.clean_unsaturated_linkers = self.clean_unsaturated_linkers
        self.defectgenerator.update_node_termination = self.update_node_termination
        self.defectgenerator.unsaturated_linkers = self.edgegraphbuilder.unsaturated_linkers
        self.defectgenerator.unsaturated_nodes = self.edgegraphbuilder.unsaturated_nodes
        #exchange
        if exchange_node_pdbfile is not None:
            self.exchange_node_pdbfile = exchange_node_pdbfile
            #use pdbreader to read the exchange node pdb files
            pdbreader = PdbReader(comm=self.comm, ostream=self.ostream)
            self.defectgenerator.exchange_node_data = pdbreader.read_pdb(
                filepath=self.exchange_node_pdbfile)
            self.defectgenerator.exchange_node_X_data = pdbreader.X_data
        if exchange_linker_pdbfile is not None:
            self.exchange_linker_pdbfile = exchange_linker_pdbfile
            #use pdbreader to read the exchange linker pdb files
            pdbreader = PdbReader(comm=self.comm, ostream=self.ostream)
            self.defectgenerator.exchange_linker_data = pdbreader.read_pdb(
                filepath=self.exchange_linker_pdbfile)
            self.defectgenerator.exchange_linker_X_data = pdbreader.X_data
        if exchange_linker_molecule is not None:
            self.exchange_linker_molecule = exchange_linker_molecule
            #use the molecule directly
            fr_ex_linker = FrameLinker(comm=self.comm, ostream=self.ostream)
            fr_ex_linker.linker_connectivity = self.linker_connectivity
            fr_ex_linker.create(molecule=self.exchange_linker_molecule)
            if self.save_files:  #TODO: check if the target directory is set
                fr_ex_linker.target_directory = self.target_directory
            fr_ex_linker.create(molecule=self.exchange_linker_molecule)
            #pass linker data
            ex_linker_center_data = fr_ex_linker.linker_center_data
            ex_linker_center_X_data = fr_ex_linker.linker_center_X_data
            if fr_ex_linker.linker_connectivity > 2:
                #recenter com of out data
                ex_linker_com = np.mean(
                    fr_ex_linker.linker_outer_X_data[:, 5:8].astype(float),
                    axis=0)
                ex_linker_outer_data = np.hstack(
                    (fr_ex_linker.linker_outer_data[:, 0:5],
                     fr_ex_linker.linker_outer_data[:, 5:8].astype(float) -
                     ex_linker_com, fr_ex_linker.linker_outer_data[:, 8:]))
                ex_linker_outer_X_data = np.hstack(
                    (fr_ex_linker.linker_outer_X_data[:, 0:5],
                     fr_ex_linker.linker_outer_X_data[:, 5:8].astype(float) -
                     ex_linker_com, fr_ex_linker.linker_outer_X_data[:, 8:]))
                ex_linker_frag_length = np.linalg.norm(
                    ex_linker_outer_X_data[0, 5:8].astype(float) -
                    ex_linker_outer_X_data[1, 5:8].astype(float))
            else:
                ex_linker_frag_length = np.linalg.norm(
                    ex_linker_center_X_data[0, 5:8].astype(float) -
                    ex_linker_center_X_data[1, 5:8].astype(float))

            self.defectgenerator.exchange_linker_data = ex_linker_center_data
            self.defectgenerator.exchange_linker_X_data = ex_linker_center_X_data

        exG = self.defectgenerator.exchange_items(exchange_indices, self.graph.copy())
        # create a new Framework object to hold the exchanged graph
        new_framework = Framework(comm=self.comm, ostream=self.ostream)
        #inherit the properties from the current framework all attributes except graph
        for attr, value in self.__dict__.items():
            if attr not in ['graph','defectgenerator','framework_data']:
                setattr(new_framework, attr, safe_copy(value))
        new_framework.graph = exG.copy()
        new_framework.get_merged_data()
        return new_framework

    def remove_defects(self, remove_indices=[]):
        self.defectgenerator = TerminationDefectGenerator(comm=self.comm,ostream=self.ostream)
        self.defectgenerator.use_termination = self.termination
        self.defectgenerator.termination_data = self.termination_data
        self.defectgenerator.termination_X_data = self.termination_X_data
        self.defectgenerator.termination_Y_data = self.termination_Y_data
        self.defectgenerator.linker_connectivity = self.linker_connectivity
        self.defectgenerator.node_connectivity = self.node_connectivity + self.vir_edge_max_neighbor if self.add_virtual_edge else self.node_connectivity
        self.defectgenerator._debug = self._debug
        self.defectgenerator.eG_index_name_dict = self.graph_index_name_dict
        self.defectgenerator.sc_unit_cell = self.sc_unit_cell
        self.defectgenerator.sc_unit_cell_inv = self.sc_unit_cell_inv
        self.defectgenerator.clean_unsaturated_linkers = self.clean_unsaturated_linkers #boolean
        self.defectgenerator.update_node_termination = self.update_node_termination #boolean
        self.defectgenerator.matched_vnode_xind = self.matched_vnode_xind
        self.defectgenerator.xoo_dict = self.xoo_dict
        self.defectgenerator.use_termination = self.termination
        self.defectgenerator.unsaturated_linkers = self.unsaturated_linkers
        self.defectgenerator.unsaturated_nodes = self.unsaturated_nodes


        #remove
        rmG = self.defectgenerator.remove_items_or_terminate(remove_indices, self.graph.copy())
        #create a new Framework object to hold the removed graph
        new_framework = Framework(comm=self.comm, ostream=self.ostream)
        #inherit the properties from the current framework all attributes except graph
        for attr, value in self.__dict__.items():
            if attr not in ['graph','defectgenerator','framework_data']:
                setattr(new_framework, attr, safe_copy(value))
        new_framework.graph = rmG.copy()
        new_framework.matched_vnode_xind = safe_copy(self.defectgenerator.updated_matched_vnode_xind)
        new_framework.unsaturated_linkers = safe_copy(self.defectgenerator.unsaturated_linkers)
        new_framework.unsaturated_nodes = safe_copy(self.defectgenerator.updated_unsaturated_nodes)
        new_framework.clean_unsaturated_linkers = self.clean_unsaturated_linkers
        new_framework.get_merged_data()
        return new_framework

    def get_merged_data(self):
        self.mofwriter.G = self.graph
        self.mofwriter.frame_cell_info = self.supercell_info
        self.mofwriter.sc_unit_cell = self.sc_unit_cell
        self.mofwriter.xoo_dict = self.xoo_dict
        self.mofwriter.dummy_atom_node_dict = self.dummy_atom_node_dict
        self.mofwriter.target_directory = self.target_directory
        self.mofwriter.supercell_boundary = self.supercell
        self.mofwriter._debug = self._debug
        self.framework_data, self.framework_fcoords_data = self.mofwriter.only_get_merged_data()
        self.residues_info = self.mofwriter.residues_info
        self.linker_molecule_data = self.mofwriter.edges_data[0] if self.mofwriter.edges_data else None


    def write(self, format=[], filename=None):
        self.filename = str(Path(filename).parent / Path(filename).stem) if filename is not None else f"{self.mof_family}_mofbuilder_output"
        self.mofwriter.filename = self.filename #whole path including directory
        self.ostream.print_info(f"Writing output files to {self.filename}.*")
        self.ostream.flush()

        if "xyz" in format:
            self.mofwriter.write_xyz(skip_merge=True)
        if "cif" in format:
            self.mofwriter.write_cif(skip_merge=False,
                                    supercell_boundary=self.supercell,
                                    frame_cell_info=self.supercell_info)
        if "pdb" in format:
            self.mofwriter.write_pdb(skip_merge=True)
        if "gro" in format:
            self.mofwriter.write_gro(skip_merge=True)
            
        ##write linker data to a file
        #with open(str(Path( self.mof_family + "_linker.xyz")), 'w') as f:
        #    for line in self.mofwriter.edges_data[0]:
        #        f.write(' '.join(map(str, line)) + '\n')


    def solvate(self, solvents_files=[], solvents_proportions=[], solvents_quantities=[], padding_angstrom=10):
        self.solvationbuilder.solvents_files = solvents_files if solvents_files else self.solvents 
        self.solvationbuilder.solute_data = self.framework_data
        self.solvationbuilder.solvents_proportions = solvents_proportions if solvents_proportions else self.solvents_proportions
        self.solvationbuilder.solvents_quantities = solvents_quantities if solvents_quantities else self.solvents_quantities
        self.solvationbuilder.target_directory = self.target_directory
        self.solvationbuilder.box_size = self.supercell_info[0:3] + np.array([padding_angstrom, padding_angstrom, padding_angstrom])
        self.solvents_dict = self.solvationbuilder.solvate()
        self.framework_data, self.solvents_data = self.solvationbuilder._update_datalines()
        #update residue info 
        def update_residues_info(solvents_dict,residue_info):
            for k, v in solvents_dict.items():
               residue_info[k]=v['accepted_quantity']
            return residue_info
        if self.solvents_dict is not None:
            self.residues_info = update_residues_info(self.solvents_dict,self.residues_info)
        self.solvation_system_data = np.vstack((self.framework_data, self.solvents_data))

        #write solvated system to gro file
        file_name = f"{self.mof_family}_in_solvent"
        self.solvationbuilder.write_output(output_file=file_name,format=["gro"])
        self.solvated_gro_file = str(Path(self.target_directory, file_name + ".gro"))

        if self.solvents is None or len(self.solvents) == 0:
            self.solvents = self.solvationbuilder.solvents_files


    def generate_linker_forcefield(self):
        self.linker_ff_gen = LinkerForceFieldGenerator(comm=self.comm, ostream=self.ostream)
        self.linker_ff_gen.target_directory = self.target_directory
        self.linker_ff_gen.linker_ff_name = self.linker_ff_name if self.linker_ff_name is not None else f"{self.mof_family}_linker"
        self.linker_ff_gen.save_files = self.save_files
        self.linker_ff_gen._debug = self._debug

        if self.provided_linker_itpfile is not None:
            self.ostream.print_info(
                "Linker force field is provided by the user, will map it directly."
            )
            self.ostream.flush()
            self.linker_ff_gen.src_linker_molecule = self.src_linker_molecule
            self.linker_ff_gen.src_linker_forcefield_itpfile = self.provided_linker_itpfile
            self.linker_ff_gen.linker_residue_name = "EDG"
            self.linker_ff_gen.map_existing_forcefield(self.mofwriter.edges_data[0])
            return

        self.linker_ff_gen.linker_optimization = self.linker_reconnect_opt
        self.linker_ff_gen.linker_residue_name="EDG"
        self.linker_ff_gen.optimize_drv = self.linker_reconnect_drv  # xtb or qm
        #self.linker_ff_gen.linker_ff_name = self.linker_ff_name if self.linker_ff_name is not None else f"{self.mof_family}_linker"
        self.linker_ff_gen.linker_charge = self.linker_charge if self.linker_charge is not None else -1 * int(
            self.linker_connectivity)
        self.linker_ff_gen.linker_multiplicity = self.linker_multiplicity if self.linker_multiplicity is not None else 1
        self.ostream.print_info(
            f"linker charge is set to {self.linker_ff_gen.linker_charge}")
        self.ostream.print_info(
            f"linker multiplicity is set to {self.linker_ff_gen.linker_multiplicity}"
        )
        self.ostream.flush()
        if self.mofwriter.edges_data:
            self.linker_ff_gen.generate_reconnected_molecule_forcefield(self.mofwriter.edges_data[0])

    def md_prepare(self):
        #write gro file for the framework
        self.generate_linker_forcefield()
        self.gmx_ff = GromacsForcefieldMerger()
        self.gmx_ff._debug = self._debug
        self.gmx_ff.solvents_dict = self.solvents_dict
        self.gmx_ff.database_dir = self.data_path if self.data_path is not None else get_data_path()
        self.gmx_ff.target_dir = self.target_directory if self.target_directory is not None else Path.cwd()
        self.gmx_ff.node_metal_type = self.node_metal
        self.gmx_ff.dummy_atom_node = self.dummy_atom_node
        self.gmx_ff.solvents_name = [str(Path(i).stem) for i in self.solvents]

        self.gmx_ff.termination_name = self.termination_name
        self.gmx_ff.linker_itp_dir = self.target_directory
        self.gmx_ff.linker_name = self.linker_ff_gen.linker_ff_name
        self.gmx_ff.residues_info = self.residues_info
        self.gmx_ff.mof_name = self.mof_family
        self.gmx_ff.generate_MOF_gromacsfile()
        if self.solvated_gro_file is None:
            self.ostream.print_warning(f"MOF system is not solvated!")
            self.ostream.flush()
            if self.filename is None:
                self.filename = str(Path(self.target_directory) / f"{self.mof_family}_mofbuilder_output")
            system_pbc = False
            #write gro file for the framework only
            self.ostream.print_info(f"Writing gro file for the framework only.")
            self.ostream.flush()
            self.ostream.print_info(f"MD input gro file will be {self.filename}.gro")
            self.ostream.flush()
            self.write(format=["gro"], filename=self.filename)
            grofile = self.filename + ".gro"
        else:
            grofile = self.solvated_gro_file
            system_pbc = True
        self.ostream.print_info(f"MD input gro file: {grofile}, top file: {self.gmx_ff.top_path}")
        self.ostream.flush()
        #setup MD driver
        self.md_driver = OpenmmSetup(gro_file=grofile, top_file=self.gmx_ff.top_path, comm=self.comm, ostream=self.ostream)
        self.md_driver.system_pbc = system_pbc
            # Run EM + NVT + NPT with single continuous PDB trajectory


    def show(self,w=800,h=600,res_id=True,res_name=True):
        self.viewer= Viewer()
        self.viewer.eG_dict= self.graph_index_name_dict
        self.viewer.merged_lines =self.framework_data
        self.viewer.lines_show(w,h,res_id,res_name)
    