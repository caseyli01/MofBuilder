import re
import sys
from pathlib import Path
from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master,hartree_in_kcalpermol, hartree_in_kjpermol
from veloxchem.errorhandler import assert_msg_critical
import mpi4py.MPI as MPI

class GromacsForcefieldMerger():

    def __init__(self, comm=None, ostream=None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream or OutputStream(sys.stdout if self.rank ==
                                               mpi_master() else None)

        #need to be set before use
        self.database_dir = None
        self.target_dir = None
        self.node_metal_type = None
        self.dummy_atom_node=False
        self.termination_name = None
        self.linker_itp_dir = ''
        self.linker_name=None
        self.residues_info = None
        self.mof_name = None
        self.other_residues=['O','HO','HHO']
    

        #self.solvate = False #later
        #
        self.solvents_name = None
        #self.neutral_system = False #later
        #self.counter_ion_names = None #later

        self._debug = False


    def _copy_file(self, old_path, new_path):
        src = Path(old_path)
        dest = Path(new_path)
        if (not dest.is_file()):
            if not dest.parent.is_dir():
                dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(src.read_text())
        if self._debug:
            self.ostream.print_info(f"File copied from {old_path} to {new_path}")
            self.ostream.flush()

    def _backup_and_rename(self,target_path: str):
        p = Path(target_path)
        if p.exists() and any(p.iterdir()):  # non-empty
            i = 1
            new_path = Path(p.parent, f"#{i}_{p.name}")
            while new_path.exists():
                i += 1
                new_path = Path(p.parent, f"#{i}_{p.name}")
            self.ostream.print_info(f"{p} existed and not empty, renaming {p} --> {new_path}")
            self.ostream.flush()
            p.rename(new_path)
            #recreate the original folder
            Path(target_path).mkdir(parents=True, exist_ok=True)

    def _get_itps_from_database(self, data_path=None):
        # itps nodes_database, edges, sol, gas
        if data_path is None:
            data_path = self.database_dir
        target_itp_path = Path(self.target_dir, 'MD_run/itps')
        #initialize itp folder
        #if the target_itp_path exist and not empty, then rename the existed folder as _old as prefix recursively
        self._backup_and_rename(str(target_itp_path))
        target_itp_path.mkdir(parents=True, exist_ok=True)

        # copy nodes itps
        if self.dummy_atom_node:
            node_itp_name = f"{self.node_metal_type}_dummy"
        else:
            node_itp_name = f"{self.node_metal_type}"
        if self._debug:
            self.ostream.print_info(f"looking for {node_itp_name}.itp for node")

        for i in Path(data_path, 'nodes_itps').rglob("*.itp"):
            #find correct node itp file, check dummy or not dummy, or other residues like O HO HHO
            if (i.stem== node_itp_name) or (i.stem in self.other_residues):
                dest_p = Path(target_itp_path,i.name)
                self._copy_file(i, dest_p)


        # copy EDGE(/TERM) itps
        if self.linker_itp_dir not in [None, '']:
            for j in Path(self.linker_itp_dir).rglob('*.itp'):
                itp_name = j.stem
                dest_p = Path(target_itp_path,j.name)
                if itp_name == Path(self.linker_name).stem:
                    self._copy_file(j, dest_p)

        # copy TERM itp
        for k in Path(data_path, 'terminations_itps').rglob('*.itp'):
            dest_p = Path(target_itp_path,k.name)
            if k.stem == Path(self.termination_name).stem:
                self._copy_file(k, dest_p)
                self.ostream.print_info(f"term.  {k} to {dest_p}")
                self.ostream.flush()

        # copy solvent, ions, gas itps
        for sol in self.solvents_name:
            src_p= Path(data_path, 'solvents_database', f'{sol}.itp')
            dest_p = target_itp_path / f'{sol}.itp'
            self._copy_file(src_p,dest_p)

        # Print target_itp_path files
        final_itp_files = [str(i) for i in Path(target_itp_path).rglob("*.itp")]
        str_itps= ",".join(final_itp_files)
        if self._debug:
            self.ostream.print_info(f"{str(target_itp_path)} directory have {len(final_itp_files)} files")
            self.ostream.print_info(f"include {str_itps}")
            self.ostream.flush()


    ########below are from top_combine.py##########

    def _itp_extract(self, itp_file):
        with open(itp_file, "r") as f:
            lines = f.readlines()
        keyword1 = "atomtypes"
        keyword2 = "moleculetype"
        start = None
        for eachline in lines:  # search for keywords and get linenumber
            if re.search(keyword1, eachline):
                start = lines.index(eachline) + 2

            elif re.search(keyword2, eachline):
                end = lines.index(eachline) - 1
        if start is None:
            return []
        target_lines = [line for line in lines[start:end] if line.strip()]

        newstart = end + 1

        with open(itp_file, "w") as fp:
            fp.writelines(lines[newstart:])

        # target_lines.append("\n")
        sec1 = target_lines
        return sec1

    def _extract_atomstypes(self, itp_path):
        all_secs = []
        for f in Path(itp_path).rglob("*itp"):
            if str(Path(f).parent) not in ["posre.itp"]:
                if self._debug:
                    self.ostream.print_info(f"found file: {f}")
                    self.ostream.flush()
                sec_atomtypes = self._itp_extract(f)
                all_secs += sec_atomtypes
        return all_secs

    def _get_unique_atomtypes(self, all_secs):
        types = [str(line.split()[0]) for line in all_secs]
        overlap_lines = []
        for ty in set(types):
            search = [ind for ind, value in enumerate(types) if value == ty]
            if len(search) > 1:
                overlap_lines += search[1:]
        unique_atomtypes = [
            all_secs[i] for i in range(len(all_secs)) if i not in overlap_lines
        ]
        return unique_atomtypes



    ####below are from itp_process.py############################


    def _parsetop(self, inputfile):
        # newpath = os.path.abspath ( '')+'/'+str(outputfile)+'/'    # input file
        # os.makedirs(newpath,exist_ok=True)
        with open(inputfile, "r") as fp:
            original_lines = fp.readlines()

        lines = [line for line in original_lines if line.strip()]
        number = []
        # includelines_number = []
        lineNumber = 1
        # lineNumber_include = 1
        keyword1 = "]"  # input('Keyword:')
        for eachline in lines:  # search for keywords and get linenumber
            m = re.search(keyword1, eachline)
            if m is not None:
                number.append(lineNumber - 1)  # split by linenumber
            lineNumber += 1

        number.append(len(lines))
        number = list(set(number))
        number.sort()
        size = int(len(number))
        # print(number)

        middlelines = []
        sectorname = []
        for i in range(size - 1):
            # set output range
            start = number[i]
            end = number[i + 1]
            middlelines.append(lines[start:end])
            sectorname.append(lines[start])

        return middlelines, sectorname


    # fetch atomtype sector

 



    def _generate_top_file(self, itp_path, data_path=None, res_info=[], model_name=None):
        all_secs = self._extract_atomstypes(itp_path)
        unique_atomtypes = self._get_unique_atomtypes(all_secs)
        middlelines, sectorname = self._parsetop(
            str(Path(data_path, "nodes_itps/template.top")))  # fetch template.top

        top_res_lines = []
        for resname in list(res_info):
            if resname[0]==';':
                continue
            if res_info[resname] <= 0:
                continue
            line = "%-5s%16d" % (resname[:3], res_info[resname])
            top_res_lines.append(line)
            top_res_lines.append("\n")

        top_itp_lines = []
        for i in Path(itp_path).rglob("*itp"):
            if str(Path(i).name) not in ["posre.itp"]:
                if self._debug:
                    self.ostream.print_info(f"found file: {i} in path {itp_path}")
                    self.ostream.flush()
                line = '#include "itps/' + i.name + '"\n'
                top_itp_lines.append(line)
                if self._debug:
                    self.ostream.print_info(f"line{line}")
                    self.ostream.flush()
        # sec1 = unique_atomtypes
        # sec2 = top_itp_lines
        # sec3 = ["MOF" + "\n" + "\n"]
        # sec4 = top_res_lines + ["\n"] + ["\n"]

        newtop = (middlelines[0] + ["\n"] + ["\n"] + middlelines[1] +
                unique_atomtypes + ["\n"] + ["\n"] + top_itp_lines + ["\n"] +
                ["\n"] + middlelines[2] + ["MOF"] + ["\n"] + ["\n"] +
                middlelines[3] + ["\n"] + top_res_lines)
        if model_name is None:
            model_name = "MOF"
        topname = model_name + ".top"
        top_path = Path(self.target_dir, "MD_run", topname)
        top_path.parent.mkdir(parents=True, exist_ok=True)

        with open(top_path, "w") as f:
            f.writelines(newtop)
        self.ostream.print_info(f" {topname} is generated")
        self.ostream.flush()
        return top_path


    def _copy_mdps(self, data_path=None):
        if data_path is None:
            data_path = self.database_dir
        dest_mdp_path = Path(self.target_dir,"MD_run","mdps")
        dest_mdp_path.mkdir(parents=True, exist_ok=True)
        src_mdp_path = Path(data_path,"mdps")
        for i in src_mdp_path.rglob("*.mdp"):
            self._copy_file(i, Path(dest_mdp_path, Path(i).name))
        return dest_mdp_path


        



    #########below are from atom2C.py##########



        
    def generate_MOF_gromacsfile(self):
        database_path = self.database_dir
        itps_path = Path(self.target_dir, 'MD_run/itps')
        res_info = self.residues_info
        model_name = self.mof_name

        self._get_itps_from_database()
        self.top_path = self._generate_top_file(itps_path,database_path,res_info,model_name)
        self._copy_mdps()


if __name__ == "__main__":
    gmx_ff = GromacsForcefieldMerger()
    gmx_ff.database_dir = 'tests/database'
    gmx_ff.target_dir = 'tests/out'
    gmx_ff.node_metal_type = 'Zr'
    gmx_ff.dummy_atom_node= True
    gmx_ff.termination_name = 'methyl'
    gmx_ff.linker_itp_dir = ''
    gmx_ff.linker_name='Linker'
    gmx_ff.residues_info = {"METAL":5}
    gmx_ff.mof_name = "testmof"
    gmx_ff.generate_MOF_gromacsfile()