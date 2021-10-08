#!/usr/bin/env python3
import numpy as np
from itertools import combinations, product

def all_states(nup, ndn, nsite):
  sites = np.arange(nsite)
  cup = combinations(sites, nup)
  cdn = combinations(sites, ndn)
  combs = product(cup, cdn)
  return combs

def bit_string(iup, idn, nsite):
  b1 = np.zeros(nsite, dtype=bool)
  b1[list(iup)] = True
  b2 = np.zeros(nsite, dtype=bool)
  b2[list(idn)] = True
  b = np.concatenate([b1, b2])
  return b

def nbit_diff(bi, bj):
  ndiff = (bi != bj).sum()
  return ndiff

def diff(bi, bj):
  diffl = np.where(bi != bj)[0]
  create = []
  destroy = []
  for idiff in diffl:
    if bj[idiff]:
      destroy.append(idiff)
    else:
      create.append(idiff)
  return create, destroy

def count_states(nup, ndn, nsite):
  combs = all_states(nup, ndn, nsite)
  n = 0
  for c in combs:
    n += 1
  return n

def show_states(nup, ndn, nsite, show_diff=True):
  nstate = count_states(nup, ndn, nsite)
  print('%d states with nup=%d ndn=%d in nmo=%d' % (nstate, nup, ndn, nsite))

  print('# <i|')
  for c in all_states(nup, ndn, nsite):
    bi = bit_string(*c, nsite)
    print(bi.astype(int))

    if show_diff:
      fmt = ' %d %24s   %s'
      print('# ndiff   (create, destroy)      |j>')
      for c1 in all_states(nup, ndn, nsite):
        bj = bit_string(*c1, nsite)
        ndiff = nbit_diff(bi, bj)
        d = diff(bi, bj)
        line = fmt % (ndiff, d, bj.astype(int))
        print(line)
      return

def main():
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('nmo', type=int)
  parser.add_argument('nup', type=int)
  parser.add_argument('ndn', type=int)
  parser.add_argument('--show_diff', action='store_true')
  args = parser.parse_args()

  show_states(args.nup, args.ndn, args.nmo, show_diff=args.show_diff)

if __name__ == '__main__':
  main()
