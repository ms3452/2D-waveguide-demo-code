Demo-code associated with out paper *Scaling on-chip photonic neural processors using arbitrarily programmable wave propagation*.
<p align="center">
<img src="https://github.com/user-attachments/assets/5a1bd570-0beb-4959-837f-6a1d0b965d23" width="800">
</p>

# How to get started

<details>
<summary>Want the <b>simplest possible code</b> to simulate a given refractuve-index pattern on a 2D-programmable waveguide?</summary>
  
  > Notebook 1 contains code that manually sets up the refractive-index distribution of a Y-splitter and simulates beam propagation through it.
</details>

<details>
<summary>Want to <b>inverse-design</b> a refractive-index pattern for a 2D-programmable waveguide?</summary>
  
  > Notebook 2 contains a minimal inverse-design example that automatically calculates a refractive-index distribution for a mode converter from Gaussian beams to Hermite-Gauss modes.
</details>

<details>
<summary>Want to see a <b>machine-learning task</b> demonstration?</summary>
  
  > Notebook 3 walks through MNIST classification with a 2D-programmable waveguide.
</details>

<details>
<summary>Want to see <b>high-dimensional matrix-vector multiplications</b> in a multimode waveguide?</summary>
  
  > Notebook 4 introduces an additional step-index multimode waveguide as a background refractive index and shows mode conversion in the waveguide with a manually defined refractive-index distribution. Notebook 5 calculates a refractive-index distribution that, embedded in a multimode waveguide, performs a desired 100x100-dimensional unitary transformation.
</details>

<details>
<summary>Want to use <b>physics-aware-training</b> with a mismatched forward- and backward-pass?</summary>
  
  > Notebook 6 contains a minimal inverse-design example with a mismatched forward- and backward-pass, similar to what we used in the optical experiments with the 2D-programmable waveguide.
</details>


<!--
Want the **simplest possible code** to simulate a given refractuve-index pattern on a 2D-programmable waveguide? 
> Notebook 1 contains code that manually sets up the refractive-index distribution of a Y-splitter and simulates beam propagation through it.

Want to **inverse-design** a refractive-index pattern for a 2D-programmable waveguide? 
> Notebook 2 contains a minimal inverse-design example that automatically calculates a refractive-index distribution for a mode converter from Gaussian beams to Hermite-Gauss modes.


Want to see a **machine-learning task** demonstration? 
> Notebook 3 walks through MNIST classification with a 2D-programmable waveguide.


Want to see **high-dimensional matrix-vector multiplications** in a multimode waveguide?
> Notebook 4 introduces an additional step-index multimode waveguide as a background refractive index and shows mode conversion in the waveguide with a manually defined refractive-index distribution. Notebook 5 calculates a refractive-index distribution that, embedded in a multimode waveguide, performs a desired 100x100-dimensional unitary transformation.


Want to use **physics-aware-training** with a mismatched forward- and backward-pass?
> Notebook 6 contains a minimal inverse-design example with a mismatched forward- and backward-pass, similar to what we used in the optical experiments with the 2D-programmable waveguide.

## Notebook 1--Simulating a simple Y-splitter

This notebook contains code that manually sets up a refractive-index distribution of a Y-splitter and simulates beam propagation through it.
<img src="https://github.com/user-attachments/assets/2fcf1d2f-ea93-4618-ad84-63733a553a79" width="600">

## Notebook 2--Minimal example of inverse-design
This notebook contains a minimal inverse-design example that automatically calculates a refractive-index distribution for a mode converter from Gaussian beams to Hermite-Gauss modes.

## Notebook 3--MNIST classification
This notebook calculates a refractive-index distribution for MNIST classification with a 2D-programmable waveguide.
<img src="https://github.com/user-attachments/assets/5a1bd570-0beb-4959-837f-6a1d0b965d23" width="900">

## Notebook 4--Mode conversion in a multimode waveguide
This notebook introduces an additional step-index multimode waveguide as a background refractive index and shows mode conversion in the waveguide with a manually defined refractive-index distribution.

## Notebook 5--Matrix-vector-multiplication in a multimode waveguide
This notebook calculates a refractive-index distribution that, embedded in a multimode waveguide, performs a desired 100x100-dimensional unitary transformation.
<img width="700" alt="MVM" src="https://github.com/user-attachments/assets/32735e82-eb4f-470c-8c75-efa1e1427744" />

## Notebook 6--Minimal example of mismatched forward-backward pass
This notebook contains a minimal inverse-design example with a mismatched forward- and backward-pass, similar to what we used in the optical experiments with the 2D-programmable waveguide.

## What happens under the hood?
<img width="700" alt="code_overview" src="https://github.com/user-attachments/assets/6b984905-5796-46fd-ba28-911f998324ac" />
-->

# How to cite this code

If you use this code in your research, please consider citing the following paper:

> Onodera, T., Stein, M. M., et al (2024). Scaling on-chip photonic neural processors using arbitrarily programmable wave propagation. *arXiv:2402.17750* https://arxiv.org/abs/2402.17750v1.

# License

The code in this repository is released under the following license:

[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

A copy of this license is given in this repository as [license.txt](https://github.com/ms3452/2D-waveguide-demo-code/blob/main/license.txt).
