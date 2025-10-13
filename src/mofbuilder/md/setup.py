from openmm.app import GromacsGroFile,GromacsTopFile, PDBFile,PDBReporter,NoCutoff, Simulation,PME,HBonds,StateDataReporter
from openmm import LangevinIntegrator,MonteCarloBarostat
from openmm.unit import atmosphere,picosecond, picoseconds, kelvin,femtoseconds,nanometer
import mpi4py.MPI as MPI
import sys
from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master

class OpenmmSetup:
    def __init__(self,  gro_file, top_file, temperature_K=300, timestep_fs=1,comm=None, ostream=None):
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream or OutputStream(sys.stdout if self.rank ==
                                               mpi_master() else None)
        self.system_pbc = True  #assume PBC for MD

        self.gro_file = gro_file
        self.top_file = top_file
        self.temperature_K = temperature_K
        self.timestep_fs = timestep_fs
        self.simulation = None

        # Load GROMACS files
        self.gro = GromacsGroFile(gro_file)
        self.top = GromacsTopFile(top_file, periodicBoxVectors=self.gro.getPeriodicBoxVectors(), includeDir='.')

        

        self.positions = self.gro.positions
        self.barostat_added = False

    def _steps_from_ps(self, ps_time):
        """Convert ps to number of steps."""
        return int(ps_time / self.timestep.value_in_unit(picoseconds))

    def run_em(self, output_prefix='em', whole_traj_file=None):
        """Energy minimization."""
        print("Starting energy minimization...")
        if self.simulation is None:
            integrator = LangevinIntegrator(self.temperature, 1/picosecond, self.timestep)
            self.simulation = Simulation(self.top.topology, self.system, integrator)
            self.simulation.context.setPositions(self.positions)
        self.simulation.minimizeEnergy()
        self.positions = self.simulation.context.getState(getPositions=True).getPositions()

        #add log
        self.simulation.reporters.append(StateDataReporter(f"{output_prefix}.log", 1,
                                                          step=True, potentialEnergy=True,
                                                          temperature=True))
        # Save minimized structure
        em_file = f"{output_prefix}_minimized.pdb"
        PDBFile.writeFile(self.simulation.topology, self.positions, open(em_file,'w'))
        print("Energy minimization done.")

        # Append to whole trajectory if requested
        if whole_traj_file:
            PDBFile.writeFile(self.simulation.topology, self.positions, open(whole_traj_file,'a'))

    def _run_md(self, mode='nvt', time_ps=100, output_prefix='traj', record_interval_ps=1, whole_traj_file=None):
        nsteps = self._steps_from_ps(time_ps)
        steps_per_frame = self._steps_from_ps(record_interval_ps)
        if mode.lower() == 'npt' and not self.barostat_added:
            barostat = MonteCarloBarostat(1*atmosphere, self.temperature)
            self.system.addForce(barostat)
            self.barostat_added = True
        if self.simulation is None:
            integrator = LangevinIntegrator(self.temperature, 1/picosecond, self.timestep)
            self.simulation = Simulation(self.top.topology, self.system, integrator)
            self.simulation.context.setPositions(self.positions)

        # Set reporters
        if whole_traj_file:
            self.simulation.reporters.append(PDBReporter(whole_traj_file, steps_per_frame))
        else:
            self.simulation.reporters.append(PDBReporter(f"{output_prefix}.pdb", steps_per_frame))

        self.simulation.reporters.append(StateDataReporter(f"{output_prefix}.log", steps_per_frame,
                                                          step=True, potentialEnergy=True,
                                                          temperature=True,
                                                          density=(mode.lower()=='npt')))

        print(f"Running {mode.upper()} for {time_ps} ps ...")
        self.simulation.step(nsteps)
        self.positions = self.simulation.context.getState(getPositions=True).getPositions()
        print(f"{mode.upper()} done.")

    def run_nvt(self, nvt_time=100, output_prefix='nvt', record_interval=1, whole_traj_file=None):
        self._run_md(mode='nvt', time_ps=nvt_time, output_prefix=output_prefix,
                     record_interval_ps=record_interval, whole_traj_file=whole_traj_file)

    def run_npt(self, npt_time=100, output_prefix='npt', record_interval=1, whole_traj_file=None):
        self._run_md(mode='npt', time_ps=npt_time, output_prefix=output_prefix,
                     record_interval_ps=record_interval, whole_traj_file=whole_traj_file)

    def run_pipeline(self, steps=['em','nvt','npt'], nvt_time=100, npt_time=100,
                     output_prefix='sim', record_interval=1, whole_traj=False):
        """Run a sequence of steps. If whole_traj=True, all frames are appended to a single PDB."""
        self.temperature = self.temperature_K*kelvin
        self.timestep = self.timestep_fs*femtoseconds
        whole_traj_file = f"{output_prefix}_whole.pdb" if whole_traj else None
        
        self.system = self.top.createSystem(nonbondedMethod=PME if self.system_pbc else NoCutoff,
                                    nonbondedCutoff=1*nanometer,
                                    constraints=HBonds,
                                    rigidWater=True)
        
        for step in steps:
            if step.lower() == 'em':
                self.run_em(output_prefix=f"{output_prefix}_em", whole_traj_file=whole_traj_file)
            elif step.lower() == 'nvt':
                self.run_nvt(nvt_time=nvt_time, output_prefix=f"{output_prefix}_nvt",
                             record_interval=record_interval, whole_traj_file=whole_traj_file)
            elif step.lower() == 'npt':
                self.run_npt(npt_time=npt_time, output_prefix=f"{output_prefix}_npt",
                             record_interval=record_interval, whole_traj_file=whole_traj_file)
            else:
                print(f"Unknown step: {step}")
