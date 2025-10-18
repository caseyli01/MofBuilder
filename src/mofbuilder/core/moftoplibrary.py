import sys
from pathlib import Path
import numpy as np
import networkx as nx
import mpi4py.MPI as MPI

from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from veloxchem.errorhandler import assert_msg_critical
from veloxchem.environment import get_data_path



class MofTopLibrary:
    def __init__(self, comm=None, ostream=None, filepath=None):
        if comm is None:
            comm = MPI.COMM_WORLD

        if ostream is None:
            if comm.Get_rank() == mpi_master():
                ostream = OutputStream(sys.stdout)
            else:
                ostream = OutputStream(None)

        # mpi information
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()

        # output stream
        self.ostream = ostream

        # clean up
        if hasattr(self, "mof_top_dict"):
            del self.mof_top_dict
        if hasattr(self, "data_path"):
            del self.data_path

        self.data_path = get_data_path()
        self.mof_top_dict = None
        self.template_directory = None
        self.mof_family = None
        self.node_connectivity = None
        self.node_metal_type = None
        self.linker_connectivity = None
        self.net_type = None

        self._debug = False

    def _read_mof_top_dict(self, data_path=None):
        if data_path is None:
            data_path = self.data_path
        if Path(data_path, "MOF_topology_dict").exists():
            mof_top_dict_path = str(Path(data_path, "MOF_topology_dict"))
            with open(mof_top_dict_path, "r") as f:
                lines = f.readlines()
            # titles = lines[0].split()
            mofs = lines[1:]
        if self._debug:
            self.ostream.print_info(f"MOF_topology_dict path: {mof_top_dict_path}")
            self.ostream.print_info(f"Got {len(mofs)} MOF families from MOF_topology_dict")
            self.ostream.print_info(f"MOF families: {[mof.split()[0] for mof in mofs]}")
            self.ostream.flush()
        mof_top_dict = {}
        for mof in mofs:
            mof_name = mof.split()[0]
            if mof_name not in mof_top_dict.keys():
                mof_top_dict[mof_name] = {
                    "node_connectivity": int(mof.split()[1]),
                    "metal": [mof.split()[2]],
                    "linker_topic": int(mof.split()[3]),
                    "topology": mof.split()[-1],
                }
            else:
                mof_top_dict[mof_name]["metal"].append(mof.split()[2])
        self.mof_top_dict = mof_top_dict


    def list_mof_family(self):
        # print mof_top_dict keys fit to screen
        if self.mof_top_dict is None:
            self._read_mof_top_dict(self.data_path)
        self.ostream.print_title("Available MOF Family:")
        for mof_family in self.mof_top_dict.keys():
            self.ostream.print_info(f" - {mof_family}")

        self.ostream.flush()


    def select_mof_family(self, mof_family):
        self.mof_family = mof_family
        self.node_connectivity = self.mof_top_dict[mof_family]["node_connectivity"]
        self.linker_connectivity = self.mof_top_dict[mof_family]["linker_topic"]
        self.net_filename = self.mof_top_dict[mof_family]["topology"] + ".cif"
        # check if template cif exists
        self.ostream.print_info(f"MOF family {mof_family} is selected")
        if mof_family not in self.mof_top_dict.keys():
            self.ostream.print_warning(f"{mof_family} not in database")
            self.ostream.print_info("please select a MOF family from below:")
            self.ostream.flush()
            self.list_mof_family()
            return
        self.ostream.print_info(f"node connectivity: {self.node_connectivity}")
        self.ostream.print_info(f"linker topic: {self.linker_connectivity}")
        self.ostream.print_info(f"available metal nodes: {self.mof_top_dict[mof_family]['metal']}")
        self.ostream.flush()
        if self.template_directory is None:
            self.template_directory = str(
                Path(self.data_path, "template_database"))  # default
            self.ostream.print_info(f"Searching template cif files in {self.template_directory}...")
            self.ostream.flush()

        template_cif_file = str(Path(self.template_directory, self.net_filename))

        if not Path(template_cif_file).exists():
            self.ostream.print_info(f"{self.net_filename} net does not exist in {self.template_directory}")
            self.ostream.print_info("please select another MOF family, or upload the template cif file")

            #TODO: set it as repository for template cif files
            self.ostream.print_info("or download the template cif files from the internet and  set it as the template directory") 
            self.ostream.flush()
            return
        else:
            self.ostream.print_info(f"{self.net_filename} is found and will be used for MOF building")
            self.ostream.flush()
            self.selected_template_cif_file = template_cif_file

    def submit_template(
        self,
        template_cif,
        mof_family,
        template_mof_node_connectivity,
        template_node_metal,
        template_linker_topic,
        overwrite=False,
    ):
        # add this item to mof_top_dict in data path
        # check if template cif exists
        assert_msg_critical(
            Path(self.data_path, "template_database").is_dir(),
            f"template_database directory {Path(self.data_path, 'template_database')} does not exist, please create it first",
        )

        assert_msg_critical(
            Path(template_cif).exists(),
            f"template cif file {template_cif} does not exist, please upload it first",
        )   

        assert_msg_critical(Path(template_cif).suffix == ".cif",
            f"template cif file {template_cif} is not a cif file, please upload a cif file",
        )   

        assert isinstance(template_mof_node_connectivity, int), "please enter an integer for node connectivity"
        assert isinstance(template_node_metal, str), "please enter a string for node metal"
        assert isinstance(template_linker_topic, int), "please enter an integer for linker topic"

        self.ostream.print_info(f"Submitting {mof_family} to the database {Path(self.data_path, 'template_database')}")
        self.ostream.print_info(f"template cif file: {template_cif}")
        self.ostream.print_info(f"node connectivity: {template_mof_node_connectivity}")
        self.ostream.print_info(f"node metal: {template_node_metal}")
        self.ostream.print_info(f"linker topic: {template_linker_topic}")
        self.ostream.print_info(f"overwrite existing mof family: {overwrite}")
        self.ostream.flush()


        if mof_family in self.mof_top_dict.keys():
            if not overwrite:
                self.ostream.print_warning(
                    f"{mof_family} already exists in the database, the template you submitted will not be used, or you can set overwrite=True to overwrite the existing template"
                )
                return

        self.mof_top_dict[mof_family] = {
            "node_connectivity": template_mof_node_connectivity,
            "metal": [template_node_metal],
            "linker_topic": template_linker_topic,
            "topology": Path(template_cif).stem,
        }
        self.ostream.print_info(f"{mof_family} is added to the database")
        self.ostream.print_info(f"{mof_family} is ready for MOF building")
        self.ostream.flush()

        # rewrite mof_top_dict file
        with open(str(Path(self.data_path, "MOF_topology_dict")),
                    "w") as fp:
            head = "MOF            node_connectivity    metal     linker_topic     topology \n"
            fp.write(head)
            for key in self.mof_top_dict.keys():
                for met in self.mof_top_dict[key]["metal"]:
                    # format is 10s for string and 5d for integer
                    line = "{:15s} {:^16d} {:^12s} {:^12d} {:^18s}".format(
                        key,
                        self.mof_top_dict[key]["node_connectivity"],
                        met,
                        self.mof_top_dict[key]["linker_topic"],
                        self.mof_top_dict[key]["topology"],
                    )
                    fp.write(line + "\n")
        self.ostream.print_info("mof_top_dict file is updated")
        self.ostream.flush()
        return str(Path(self.data_path, "template_database", template_cif))
    
    def fetch(self,mof_family=None):
        self._read_mof_top_dict(self.data_path)
        if mof_family is None:
            self.ostream.print_info("please select a MOF family from below:")
            self.ostream.flush()
            self.list_mof_family()
            return 
        else:
            if mof_family not in self.mof_top_dict.keys():
                self.ostream.print_warning(f"{mof_family} not in database")
                self.ostream.print_info("please select a MOF family from below:")
                self.ostream.flush()
                self.list_mof_family()
                return 
            else:
                self.select_mof_family(mof_family)
                return self.selected_template_cif_file



if __name__ == "__main__":
    moflib = MofTopLibrary()
    moflib.data_path = 'tests/database'
    #moflib._debug = True
    moflib.fetch(mof_family="Ui-66")

    #moflib.submit_template(
    #    template_cif="tests/database/template_database/fcu.cif",
    #    mof_family="MOF-5",
    #    template_mof_node_connectivity=4,
    #    template_node_metal="Zn",
    #    template_linker_topic=2,
    #    overwrite=True,
    #)

    