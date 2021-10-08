#!/usr/bin/env python3
import numpy as np

def main():
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('fascii', type=str)
  parser.add_argument('--fh5', '-o', type=str, default='ham.h5')
  parser.add_argument('--verbose', action='store_true')
  args = parser.parse_args()
  assert args.fh5 != args.fascii

  from pyscf.tools import fcidump
  ret = fcidump.read(args.fascii)
  assert np.allclose(ret['H1'].imag, 0)
  assert np.allclose(ret['H2'].imag, 0)
  h0 = ret['ECORE']
  h1 = ret['H1'].real
  eri = ret['H2'].real
  nmo = ret['NORB']
  from pyscf.ao2mo import restore
  h2 = restore(1, eri, nmo)
  vijkl = h2.transpose(0, 2, 1, 3)

  import h5py
  with h5py.File(args.fh5, 'w') as f:
    for key, val in zip(['h0', 'h1', 'h2'], [h0, h1, vijkl]):
      f[key] = val

if __name__ == '__main__':
  main()  # set no global variable
