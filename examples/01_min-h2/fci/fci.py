#!/usr/bin/env python3
import yaml
import numpy as np
from pyscf.tools import fcidump
from pyscf import fci

def count_states(nelec, nmo):
  from scipy.special import comb
  nup, ndn = nelec
  nstate = comb(nmo, nup)*comb(nmo, ndn)
  return int(nstate)

def get_few_states(nstate, h1, eri, nmo, nelec):
  cisolver = fci.SCI()
  cisolver.max_cycle = 100
  cisolver.conv_tol = 1e-8
  cisolver.verbose = 5
  cisolver.nroots = nstate
  e, fcivec = cisolver.kernel(h1, eri, nmo, nelec)
  return e, fcivec

def get_all_states(h1, eri, nmo, nelec, save_hfci=False, use_gpu=False):
  nstate = count_states(nelec, nmo)
  idx, H_fci = fci.direct_spin1.pspace(h1, eri, nmo, nelec, np=nstate)
  if save_hfci:
    from qharv.reel import config_h5
    config_h5.saveh5('h1.h5', H_fci)
  if use_gpu:
    import cupy
    from cupy.linalg import eigvalsh
    H_fci = cupy.asarray(H_fci)
  else:
    from numpy.linalg import eigvalsh
  evals = eigvalsh(H_fci)
  #evals, evecs = np.linalg.eigh(H_fci)
  return evals

def main():
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('fascii', type=str)
  parser.add_argument('nup', type=int)
  parser.add_argument('ndn', type=int)
  parser.add_argument('--fyml', type=str)
  parser.add_argument('--lam', type=float, default=1.0)
  parser.add_argument('--e2e', type=float, default=1.0)
  parser.add_argument('--verbose', '-v', action='store_true')
  args = parser.parse_args()
  nup = args.nup
  ndn = args.ndn
  fyml = args.fyml
  if fyml is None:
    fyml = 'evals-l%f-e%f-nup%d-ndn%d.yml' % (args.lam, args.e2e, nup, ndn)

  fascii = args.fascii
  ret = fcidump.read(fascii)
  assert np.allclose(ret['H1'].imag, 0)
  assert np.allclose(ret['H2'].imag, 0)
  h0 = ret['ECORE']
  h1 = ret['H1'].real*args.lam
  eri = ret['H2'].real*args.e2e
  ntot = ret['NELEC']
  ms2 = ret['MS2']
  nelec = ((ntot+ms2)//2, ntot-(ntot+ms2)//2)
  nmo = ret['NORB']
  if (nup != nelec[0]) or (ndn != nelec[1]):
    msg = 'overwriting FCIDUMP MS2 %s' % str(nelec)
    nelec = (nup, ndn)
    msg += ' with %s' % str(nelec)
    if args.verbose:
      print(msg)

  #nstate = 5
  #evals = get_few_states(nstate, h1, eri, nmo, nelec)

  evals = get_all_states(h1, eri, nmo, nelec)
  e0 = np.sort(evals) + h0
  msg = 'ground state energy %f out of %d states\n' % (e0[0], len(evals))
  msg += ' gap = %f' % (e0[1]-e0[0])
  if args.verbose:
    print(msg)
  # save eigenvalues
  evd = dict(
    nmo = nmo,
    nup = nup,
    ndn = ndn,
    evals = e0.tolist(),
  )
  with open(fyml, 'w') as f:
    yaml.dump(evd, f)

if __name__ == '__main__':
  main()  # set no global variable
