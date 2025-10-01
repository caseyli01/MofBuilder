import sys
import mpi4py.MPI as MPI
from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from ..io.xyz_writer import XyzWriter
class Viewer:
    def __init__(self, comm=None, ostream=None, filepath=None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()

        if ostream is None:
            ostream = OutputStream(sys.stdout if self.rank ==
                                   mpi_master() else None)
        self.ostream = ostream
        self.eG_dict = None
        self.merged_lines=None

    def _reverse_eG_dict(self):

        #the eG dict is a dictionary, key is index number and value is the name need to fetch the key
        self.eG_name_idx_dict={v:int(k) for k,v in self.eG_dict.items()}
        
    def lines_show(self, w=800, h=600, res_id=True, res_name=True):
        try:
            import py3Dmol
            merged_lines = self.merged_lines
            viewer = py3Dmol.view(width=w, height=h)
            xyz_writer = XyzWriter(comm=self.comm, ostream=self.ostream)
            xyz_lines = xyz_writer.get_xyzlines(lines=merged_lines)
            viewer.addModel("".join(xyz_lines), "xyz")
            viewer.setViewStyle({"style": "outline", "width": 0.05})
            viewer.setStyle({"stick": {}, "sphere": {"scale": 0.20}})
            if res_id or res_name:
                self._reverse_eG_dict()
                old_resnumber = 0
                for line in merged_lines:
                    value_resname = line[3].split('_')[0][:3]
                    if value_resname.strip() == "TNO":
                        continue
                    value_resnumber = self.eG_name_idx_dict[line[10]]
                    if value_resnumber==old_resnumber:
                        continue

                    old_resnumber=value_resnumber
                    
                    value_x = float(line[5])
                    value_y = float(line[6])
                    value_z = float(line[7])

                    text = ""
                    if res_name:
                        text += str(value_resname)
                    if res_id:
                        text += str(value_resnumber)

                    viewer.addLabel(
                        text,
                        {
                            "position": {
                                "x": value_x,
                                "y": value_y,
                                "z": value_z,
                            },
                            "alignment": "center",
                            "fontColor": "white",
                            "font": "Arial",
                            "fontSize": 12,
                            "backgroundColor": "black",
                            "backgroundOpacity": 0.5,
                        },
                    )
            viewer.render()
            viewer.zoomTo()
            viewer.show()
        except ImportError:
            raise ImportError("Unable to import py3Dmol")
