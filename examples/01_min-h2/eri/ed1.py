#!/usr/bin/env python3
import numpy as np
from time import time
from itertools import combinations, product

import sys
sys.path.insert(0, '../../00_states/eri')
from ed0 import all_states, nbit_diff, diff, count_states, show_states

def read_h5(fh5):
  import h5py
  result = dict()
  with h5py.File(fh5, 'r') as f:
    for key in f.keys():
      result[key] = f[key][()]
  return result

def spin_and_orb(m, nmo):
  return divmod(m, nmo)

def calc_h1_ndiff0(soccl, h1):
  h1v = sum([h1[m1, m1] for (ms, m1) in soccl])
  return h1v

def calc_h2_ndiff0(soccl, eri):
  # direct - exchange
  # direct: destroy m, create m; destroy n, create n
  # exchange: destroy n, create m; destroy m, create n
  h2v = 0  # [mm|nn] - [mn|nm]
  for (ms, m1), (ns, n1) in combinations(soccl, 2):
    h2v += eri[m1, n1, m1, n1]
    # exchange contribution if same spin
    if (ms == ns):
      h2v -= eri[m1, n1, n1, m1]
  return h2v

def calc_h2_ndiff4(create, destroy, nsite, eri):
  h2v = 0  # [mp|nq] - [mq|np]
  m, n = sorted(create)
  p, q = sorted(destroy)
  ms, m1 = spin_and_orb(m, nsite)
  ps, p1 = spin_and_orb(p, nsite)
  ns, n1 = spin_and_orb(n, nsite)
  qs, q1 = spin_and_orb(q, nsite)
  if (ms == ps) and (ns == qs):
    h2v = eri[m1, n1, p1, q1]
    if (ms == ns):
      h2v -= eri[m1, n1, q1, p1]
  return h2v

def store_all_states(nup, ndn, nmo, mgb=16, verbose=True):
  from ed0 import bit_string
  nstate = count_states(nup, ndn, nmo)
  nbits = nstate*nmo*2
  GB = nbits/8/1024**3
  if verbose:
    print('storing all %d states requires %f GB' % (nstate, GB))
  if GB > mgb:
    msg = ' more than %d GB' % mgb
    raise RuntimeError(msg)
  start = time()
  states = [bit_string(*s, nmo) for s in all_states(nup, ndn, nmo)]
  end = time()
  elapsed = end-start
  if verbose:
    print('stored all states in %f seconds' % elapsed)
  return states

def build_hfci(states, h1, eri, mgb=256, verbose=True):
  nstate = len(states)
  nsite = len(h1)
  GB_ham = nstate*nstate*128/8/1024**3
  if verbose:
    print('storing FCI hamiltonian requires %f GB' % GB_ham)
  if GB_ham > mgb:
    msg = 'ham too big'
    raise RuntimeError(msg)
  start = time()
  ham = np.zeros([nstate, nstate], dtype=np.complex128)

  for istate, bi in enumerate(states):
    for jstate, bj in enumerate(states):
      ndiff = nbit_diff(bi, bj)
      # easiest cases first
      if ndiff > 4:
        continue
      h1v = 0
      h2v = 0
      # implement these cases one at a time
      #  (0up, 1dn), (1up, 1dn); (0up, 2dn); (2up, 2dn)
      #  first set eri to 0 (e2e=0), then set to 1
      if ndiff == 0:
        occl = np.where(bj)[0]
        soccl = [spin_and_orb(i, nsite) for i in occl]
        h1v = calc_h1_ndiff0(soccl, h1)
        h2v = calc_h2_ndiff0(soccl, eri)
        ham[istate, jstate] = h1v+h2v
        continue
      create, destroy = diff(bi, bj)
      if ndiff == 2:
        pass
      elif ndiff == 4:
        h2v = calc_h2_ndiff4(create, destroy, nsite, eri)
      else:
        pass  # h1v, h2v already initialized to 0
      ham[istate, jstate] = h1v+h2v

  end = time()
  elapsed = end - start
  if verbose:
    print('built hamiltonian in %f seconds' % elapsed)
  return ham

def get_all_evals(ham, verbose=True):
  from numpy.linalg import eigvalsh
  if verbose:
    print('diagonalizing...')
  start = time()
  evals = eigvalsh(ham)
  end = time()
  elapsed = end - start
  if verbose:
    print('diagonalized in %f seconds' % elapsed)
  return evals

def ed(build_hfci_func, nup, ndn, h1, eri,
       save_hfci=False, verbose=True):
  nsite = len(eri)
  assert len(h1) == nsite
  states = store_all_states(nup, ndn, nsite)
  ham = build_hfci_func(states, h1, eri, verbose=verbose)
  # check/store FCI hamiltonian
  is_hermitian = np.allclose(ham, ham.conj().T)
  if not is_hermitian:
    msg = 'FCI hamiltonian is not hermitian'
    raise RuntimeError(msg)
  if save_hfci:
    from qharv.reel import config_h5
    config_h5.saveh5('mham.h5', ham)
  # diagonalize
  evals = get_all_evals(ham, verbose=verbose)
  return evals

def write_evals(fout, nup, ndn, h1, eri, h0=0):
  import yaml
  from time import time
  nsite = len(eri)

  start = time()
  evals = ed(build_hfci, nup, ndn, h1, eri)
  end = time()
  elapsed = end-start
  print('nup=%d total time %f seconds\n' % (nup, elapsed))

  elist = (h0+evals).tolist()
  evd = dict(
    nmo = nsite,
    nup = nup,
    ndn = ndn,
    evals = sorted(elist)
  )
  with open(fout, 'w') as f:
    yaml.dump(evd, f)

def main():
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('fh5', type=str)
  parser.add_argument('nup', type=int)
  parser.add_argument('ndn', type=int)
  parser.add_argument('--lam', type=float, default=1.0)
  parser.add_argument('--e2e', type=float, default=1.0)
  parser.add_argument('--fyml', '-o', type=str)
  parser.add_argument('--show_states', action='store_true')
  parser.add_argument('--show_diff', action='store_true')
  args = parser.parse_args()
  nup = args.nup
  ndn = args.ndn

  fyml = args.fyml
  if fyml is None:
    fyml = 'evals-l%f-e%f-nup%d-ndn%d.yml' % (args.lam, args.e2e, nup, ndn)

  ret = read_h5(args.fh5)
  h0 = ret['h0'].real if 'h0' in ret else 0
  h1 = ret['h1']*args.lam
  eri = ret['h2']*args.e2e  # physicist

  nsite = len(h1)
  if args.show_states:
    show_states(nup, ndn, nsite, show_diff=args.show_diff)
    import sys
    sys.exit(0)

  write_evals(fyml, nup, ndn, h1, eri, h0=h0)

if __name__ == '__main__':
  main()
