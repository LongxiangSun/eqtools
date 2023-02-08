# eqtools

---

**A set of tools related to the forward and inverse *earthquake cycle*.**

---

## Postseismic

- pscmp_visco_postseismic.py
  - Simple calculating codes about ***Combined model of viscoelastic relaxation and stress-driven afterslip***
- Deformation components
  - CV (Viscoelastic relaxation due to coseismic): Function ***cv_cum(t, tobs, disp)***
  - AS (Afterslip): Function ***as_cum(t, disp0, tau_as=0.25, alpha=1.0)***
  - AV (Viscoelastic relaxation due to afterslip): Function ***calAS_AV(pscmpts, obsdate, eqdate, alpha, tau, unit='m', intp_tunit='Y', onlyAV=True, mcpu=4)***

## Interseismic

- pscmp_visco_interseismic.py
