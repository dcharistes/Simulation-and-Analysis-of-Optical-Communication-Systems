import numpy as np
from optic.models.devices import mzm, photodiode, edfa
from optic.models.channels import linearFiberChannel
from optic.comm.modulation import modulateGray
from optic.dsp.core import upsample, pulseShape, lowPassFIR, pnorm, signal_power
from optic.utils import parameters, dBm2W
from optic.plot import eyediagram
import matplotlib.pyplot as plt
from scipy.special import erfc
from tqdm.notebook import tqdm
import scipy as sp
from IPython.core.display import HTML
from IPython.core.pylabtools import figsize
try:
   from optic.dsp.coreGPU import checkGPU
   if checkGPU():
      from optic.dsp.coreGPU import firFilter
   else:
      from optic.dsp.core import firFilter
except ImportError:
   from optic.dsp.core import firFilter

HTML("""
<style>
.output_png {
display: table-cell;
text-align: center;
vertical-align: middle;
}
</style>
""")
figsize(10, 3)

np.random.seed(seed=123)
SpS = 16
M = 2
Rs = 1e9
Fs = Rs*SpS
Ts = 1/Fs
Pi_dBm = 0
Pi = dBm2W(Pi_dBm)

paramMZM = parameters()
paramMZM.Vpi = 2
paramMZM.Vb = -paramMZM.Vpi/2
bitsTx = np.random.randint(2, size=100000)
symbTx = modulateGray(bitsTx, M, 'pam')
symbTx = pnorm(symbTx)

symbolsUp = upsample(symbTx, SpS)
pulse = pulseShape('nrz', SpS)
pulse = pulse/max(abs(pulse))
sigTx = firFilter(pulse, symbolsUp)
Ai = np.sqrt(Pi)
sigTxo = mzm(Ai, sigTx, paramMZM)
print('Average power of the modulated optical signal [mW]: %.3fmW'%(signal_power(sigTxo)/1e-3))
print('Average power of the modulated optical signal [dBm]: %.3fdBm'%(10*np.log10(signal_power(sigTxo)/1e-3)))

fig, axs = plt.subplots(1, 2, figsize=(16,3))
interval = np.arange(16*20,16*50)
t = interval*Ts/1e-9
# plot psd
axs[0].set_xlim(-3*Rs,3*Rs);
axs[0].set_ylim(-180,-80);
axs[0].psd(sigTx,Fs=Fs, NFFT = 16*1024, sides='twosided',
label = 'RF signal spectrum')
axs[0].legend(loc='upper left');
axs[1].plot(t, sigTx[interval], label = 'RF binary signal',
linewidth=2)
axs[1].set_ylabel('Amplitude (a.u.)')
axs[1].set_xlabel('Time (ns)')
axs[1].set_xlim(min(t),max(t))
axs[1].legend(loc='upper left')
axs[1].grid()
fig, axs = plt.subplots(1, 2, figsize=(16,3))
axs[0].set_xlim(-3*Rs,3*Rs);
axs[0].set_ylim(-230,-130);
axs[0].psd(np.abs(sigTxo)**2, Fs=Fs, NFFT = 16*1024,
sides='twosided', label = 'Optical signal spectrum')
axs[0].legend(loc='upper left');

axs[1].plot(t, np.abs(sigTxo[interval])**2, label = 'Optical modulated signal', linewidth=2)
axs[1].set_ylabel('Power (p.u.)')
axs[1].set_xlabel('Time (ns)')
axs[1].set_xlim(min(t),max(t))
axs[1].legend(loc='upper left')
axs[1].grid()

paramCh = parameters()
paramCh.L = 90
paramCh.α = 0.2
paramCh.D = 16
paramCh.Fc = 193.1e12
paramCh.Fs = Fs
sigCh = linearFiberChannel(sigTxo, paramCh)

# receiver pre-amplifier
paramEDFA = parameters()
paramEDFA.G = paramCh.α*paramCh.L
paramEDFA.NF = 4.5
paramEDFA.Fc = paramCh.Fc
paramEDFA.Fs = Fs
sigCh = edfa(sigCh, paramEDFA)

paramPD = parameters()
paramPD.ideal = True
paramPD.Fs = Fs
I_Tx = photodiode(sigTxo.real, paramPD)
paramPD = parameters()
paramPD.ideal = False
paramPD.B = Rs
paramPD.Fs = Fs
I_Rx = photodiode(sigCh, paramPD)

discard = 100
eyediagram(I_Tx[discard:-discard], I_Tx.size-2*discard, SpS, plotlabel='signal at Tx', ptype='fancy')
eyediagram(I_Rx[discard:-discard], I_Rx.size-2*discard, SpS, plotlabel='signal at Rx', ptype='fancy')
# +
I_Rx = I_Rx/np.std(I_Rx)
# capture samples in the middle of signaling intervals
I_Rx = I_Rx[0::SpS]

I1 = np.mean(I_Rx[bitsTx==1])
I0 = np.mean(I_Rx[bitsTx==0])
σ1 = np.std(I_Rx[bitsTx==1])
σ0 = np.std(I_Rx[bitsTx==0])
Id = (σ1*I0 + σ0*I1)/(σ1 + σ0)
Q = (I1-I0)/(σ1 + σ0)

print('I0 = %.2f '%(I0))
print('I1 = %.2f '%(I1))
print('σ0 = %.2f '%(σ0))
print('σ1 = %.2f '%(σ1))
print('Optimal decision threshold Id = %.2f '%(Id))
print('Q = %.2f \n'%(Q))

bitsRx = np.empty(bitsTx.size)
bitsRx[I_Rx> Id] = 1
bitsRx[I_Rx<= Id] = 0
discard = 100
err = np.logical_xor(bitsRx[discard:bitsRx.size-discard], bitsTx[discard:bitsTx.size-discard])
BER = np.mean(err)

Pb = 0.5*erfc(Q/np.sqrt(2))
print('Number of counted errors = %d '%(err.sum()))
print('BER = %.2e '%(BER))
print('Pb = %.2e '%(Pb))
err = err*1.0
err[err==0] = np.nan

plt.plot(err,'o', label = 'bit errors')
plt.vlines(np.where(err>0), 0, 1)
plt.xlabel('bit position')
plt.ylabel('counted error')
plt.legend()
plt.grid()
plt.ylim(0, 1.5)
plt.xlim(0,err.size);

SpS = 16
M = 2
Rs = 40e9
Fs = SpS*Rs
Ts = 1/Fs

paramMZM = parameters()
paramMZM.Vpi = 2
paramMZM.Vb = -paramMZM.Vpi/2
pulse = pulseShape('nrz', SpS)
pulse = pulse/max(abs(pulse))

paramPD = parameters()
paramPD.ideal = False
paramPD.B = 1.1*Rs
paramPD.Fs = Fs
powerValues = np.arange(-30,-14)
BER = np.zeros(powerValues.shape)
Pb = np.zeros(powerValues.shape)

discard = 100
for indPi, Pi_dBm in enumerate(tqdm(powerValues)):
  Pi = dBm2W(Pi_dBm+3)
  bitsTx = np.random.randint(2, size=10**6)
  n = np.arange(0, bitsTx.size)

  symbTx = modulateGray(bitsTx, M, 'pam')
  symbTx = pnorm(symbTx)
  symbolsUp = upsample(symbTx, SpS)
  sigTx = firFilter(pulse, symbolsUp)

  Ai = np.sqrt(Pi)
  sigTxo = mzm(Ai, sigTx, paramMZM)
  I_Rx = photodiode(sigTxo.real, paramPD)
  I_Rx = I_Rx/np.std(I_Rx)
  I_Rx = I_Rx[0::SpS]

  I1 = np.mean(I_Rx[bitsTx==1])
  I0 = np.mean(I_Rx[bitsTx==0])
  σ1 = np.std(I_Rx[bitsTx==1])
  σ0 = np.std(I_Rx[bitsTx==0])
  Id = (σ1*I0 + σ0*I1)/(σ1 + σ0)
  Q = (I1-I0)/(σ1 + σ0)

  bitsRx = np.empty(bitsTx.size)
  bitsRx[I_Rx> Id] = 1
  bitsRx[I_Rx<= Id] = 0
  err = np.logical_xor(bitsRx[discard:bitsRx.size-discard], bitsTx[discard:bitsTx.size-discard])
  BER[indPi] = np.mean(err)
  Pb[indPi] = 0.5*erfc(Q/np.sqrt(2))

plt.figure()
plt.plot(powerValues, np.log10(Pb),'--',label='Pb (theory)')
plt.plot(powerValues, np.log10(BER),'o',label='BER')
plt.grid()
plt.ylabel('log10(BER)')
plt.xlabel('Pin (dBm)');
plt.title('Bit-error performance vs input power at the pin receiver')
plt.legend();
plt.ylim(-10,0);
plt.xlim(min(powerValues), max(powerValues));

