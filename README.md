# eried
Exact Diagonalization (ED) given Electron Repusion Integrals (ERI)

WARNING
====
This inefficient Python implementation is NOT meant to be a production code.
I view it as an executable introduction for someone new to ED.
A series of simple examples guides one's implementation of Slater-Condon
 rules one at a time.

## Definitions

Consider the electronic Hamiltonian having decoupled spin sectors
<img src="https://render.githubusercontent.com/render/math?math=\hat{H}=H_0%2B\hat{H}_1%2B\hat{H}_2,">
where <img src="https://render.githubusercontent.com/render/math?math=H_0"> is a constant energy shift independent of the electronic degrees of freedom, e.g.  ion-ion interaction energy. <img src="https://render.githubusercontent.com/render/math?math=\hat{H}_1"> is 1-body interaction, e.g. electron-ion interaction.  <img src="https://render.githubusercontent.com/render/math?math=\hat{H}_2"> is 2-body interaction.

Given a basis of single-particle orbitals, this collinear-spin electronic Hamiltonian can be written as
<img src="https://render.githubusercontent.com/render/math?math=H_0%2B
\sum_{\sigma=\alpha}^\beta\sum_{i,j=1}^{N_\sigma} h_{ij}c_{i\sigma}^\dagger c_{j\sigma}%2B
\frac{1}{2}\sum_{\sigma,\sigma'=\alpha}^\beta\sum_{p,q,r,s=1}^{N_\sigma} v_{pqrs} c_{p\sigma}^\dagger c_{q\sigma'}^\dagger c_{s\sigma} c_{r\sigma}.
">
Here <img src="https://render.githubusercontent.com/render/math?math=h_{ij}=\langle i|\hat{H}_1|j\rangle">. The ERIs indices are ordered by physicist's notation <img src="https://render.githubusercontent.com/render/math?math=v_{pqrs}=\langle pq|rs \rangle">.
Notice, the order of the creation and destruction operators is pqsr!

The goal of this code is to diagonalize H given H0, H1, H2.

Example 01: minimum basis H2
----

WARNING: This example is deceivingly simple, because
 h1 is diagonal AND
 there is no need to consider anti-commuting operators.

Step 1: implement `calc_h1_ndiff0` (H1 ndiff0)
 pass --e2e 0, 0up1dn and 1up1dn

Step 2: implement `calc_h2_ndiff0` (H2 ndiff0)
 pass --lam 0, 0up2dn

Step 3: implement `calc_h2_ndiff0` (H2 ndiff4)
 pass 1up1dn

Example 02: minimum basis H4
----

Check: pass --e2e 0&1, 0up1dn

Step 1: implement `calc_h1_ndiff2` (H1 ndiff2)
 pass --e2e 0, 0up1dn and 1up1dn

Step 2: implement `calc_h2_ndiff2` (H2 ndiff2)
 pass 1up1dn

Step 3: implement `calc_permutation_sign`
 pass 0up2dn

Step 4: implement `calc_p2`
 pass 1up2dn

ED code should be correct after this point.
