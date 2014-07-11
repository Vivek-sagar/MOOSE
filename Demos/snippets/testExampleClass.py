import sys
sys.path.append('../../python') # in case we do not have moose/python in PYTHONPATH
import pylab
import moose
# Create the somatic compartment
model = moose.Neutral('/model') # This is a container for the model

ex = moose.Example('/model/ex')
ex.y = 15

# Setup data recording
data = moose.Neutral('/data')
Tab = moose.Table('/data/axon_Vm')
#moose.connect(axon_Vm, 'requestOut', axon, 'getVm')
moose.connect(Tab, 'requestOut', ex, 'getX')

# Now schedule the sequence of operations and time resolutions
moose.setClock(0, 0.025e-3)
moose.setClock(1, 0.025e-3)
moose.setClock(2, 0.25e-3)
# useClock: First argument is clock no.
# Second argument is a wildcard path matching all elements of type Compartment
# Last argument is the processing function to be executed at each tick of clock 0 
moose.useClock(0, '/model/ex', 'init')
moose.useClock(1, '/model/ex', 'process')
moose.useClock(2, Tab.path, 'process')
# Now initialize everything and get set
moose.reinit()

moose.start(50e-3)
clock = moose.Clock('/clock') # Get a handle to the global clock
pylab.plot(pylab.linspace(0, clock.currentTime, len(Tab.vector)), Tab.vector, label='')
pylab.legend()
pylab.show()

