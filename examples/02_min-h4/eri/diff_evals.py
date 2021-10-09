#!/usr/bin/env python3
import yaml
import numpy as np

def read_evals(fyml):
  with open(fyml, 'r') as f:
    evd = yaml.safe_load(f)
  elist = evd['evals']
  return np.array(elist)

def main():
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument('nup', type=int)
  parser.add_argument('ndn', type=int)
  parser.add_argument('--lam', type=float, default=1)
  parser.add_argument('--e2e', type=float, default=1)
  parser.add_argument('--tol', type=float, default=1e-12)
  parser.add_argument('--verbose', action='store_true')
  args = parser.parse_args()
  nup = args.nup
  ndn = args.ndn

  prefix = 'evals-l%f-e%f-nup%d-ndn%d' % (args.lam, args.e2e, nup, ndn)
  fyml0 = '../fci/%s.yml' % prefix
  fyml1 = '../eri/%s.yml' % prefix

  e0 = read_evals(fyml0)
  e1 = read_evals(fyml1)
  de = e1-e0
  sel = abs(de) > args.tol
  idx = np.where(sel)[0]
  print(idx)
  print(de[sel])

if __name__ == '__main__':
  main()  # set no global variable
