* version 107 direct OSDI probe
.include /home/ZhangLexin/PALS/wmpc/results/bsimcmg_version_gate_20260903/modelcard.nmos.v107
.include /home/ZhangLexin/PALS/wmpc/results/bsimcmg_version_gate_20260903/modelcard.pmos.v107
VDD supply 0 0.7
VIN vi 0 0.35
Np1 vo vi supply 0 pmos1 L=30n HFIN=30n TFIN=15n NFIN=10
Nn1 vo vi 0 0 nmos1 L=30n HFIN=30n TFIN=15n NFIN=10
Rload vo 0 1meg
.control
pre_osdi /home/ZhangLexin/PALS/wmpc-gt2n-compat/results/gt2n_ngspice_compat_20260903/toolchain_check/bsim_versions/bsimcmg_v107.osdi
op
print v(vo)
.endc
.end
