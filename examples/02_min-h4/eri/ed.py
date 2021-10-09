#!/usr/bin/env python3
import numpy as np
from time import time
from itertools import combinations, product

import sys
sys.path.insert(0, '../../00_states/eri')
from ed0 import all_states, nbit_diff, diff, count_states, show_states
sys.path.insert(0, '../../01_min-h2/eri')
from ed1 import read_h5, spin_and_orb, calc_h1_ndiff0
from ed1 import calc_h2_ndiff0, calc_h2_ndiff4
from ed1 import ed

def calc_h1_ndiff2(M, P, h1):
  ms, m1 = M
  ps, p1 = P
  return h1[m1, p1]

def calc_h2_ndiff2(M, P, soccl, eri):
  ms, m1 = M
  ps, p1 = P
  h2v = 0  # [mp|nn] - [mn|np]
  for (ns, n1) in soccl:
    h2v += eri[m1, n1, p1, n1]
    if ms == ns:
      h2v -= eri[m1, n1, n1, p1]
  return h2v

def calc_permutation_sign(m, p, bj_in):
  bj = bj_in.copy()
  ncp = sum(bj[:p])
  bj[p] = False
  ncm = sum(bj[:m])
  return (-1)**(ncp+ncm)

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
        m = create[0]
        p = destroy[0]
        M = spin_and_orb(m, nsite)
        P = spin_and_orb(p, nsite)
        if M[0] == P[0]:
          mybj = bj[M[0]*nsite:(M[0]+1)*nsite]
          pm = calc_permutation_sign(M[1], P[1], mybj)
          h1v = pm*calc_h1_ndiff2(M, P, h1)
          occ2l = np.where(bi&bj)[0]
          socc2l = [spin_and_orb(i, nsite) for i in occ2l]
          h2v = pm*calc_h2_ndiff2(M, P, socc2l, eri)
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
