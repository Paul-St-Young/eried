#!/usr/bin/env python3
import numpy as np

def main():
  rb = 0.74
  from pyscf import gto, scf
  mol = gto.M(
    atom = 'H 0 0 0; H 0 0 %f' % rb,
    basis = 'sto-3g',
  )
  mf = scf.RHF(mol)
  mf.chkfile = 'scf.h5'
  mf.run()

  from pyscf.tools import fcidump
  fcidump.from_scf(mf, 'FCIDUMP')

if __name__ == '__main__':
  main()  # set no global variable
