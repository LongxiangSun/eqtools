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

### Reference

[1] Diao, F., R. Wang, Y. Wang, X. Xiong, and T. R. Walter (2018), Fault behavior and lower crustal rheology inferred from the first seven years of postseismic GPS data after the 2008 Wenchuan earthquake, Earth Planet. Sci. Lett., 495, 202-212, doi:10.1016/j.epsl.2018.05.020.

[2] Diao, F., R. Wang, X. Xiong, and C. Liu (2021), Overlapped Postseismic Deformation Caused by Afterslip and Viscoelastic Relaxation Following the 2015 Mw 7.8 Gorkha (Nepal) Earthquake, J. Geophys. Res.: Sol. Ea., 126(3), doi:10.1029/2020jb020378.

[3] He Ke., Y. Wen, C. Xu, W. Xiong, J. Zang (2023), Afterslip and viscoelastic relaxation following the 2021 *Mw* 7.4 Maduo earthquake inferred from InSAR and GPS observations, In Review.

## Interseismic

- pscmp_visco_interseismic.py
  - Calculating velocity related to ***earthquake cycle***
- Main function
  - ***calviscoGfromPscmp***(pscmpts, T=None, diffint=None, unit='m')

### Reference

[1] Diao, F., X. Xiong, R. Wang, T. R. Walter, Y. Wang, and K. Wang (2019), Slip Rate Variation Along the Kunlun Fault (Tibet): Results From New GPS Observations and a Viscoelastic Earthquake‐Cycle Deformation Model, Geophys. Res. Lett., 46(5), 2524-2533, doi:10.1029/2019gl081940.

[2] Zhu, Y., K. Wang, and J. He (2020), Effects of Earthquake Recurrence on Localization of Interseismic Deformation Around Locked Strike‐Slip Faults, J. Geophys. Res.: Sol. Ea., 125(8), doi:10.1029/2020jb019817.

[3] Wang, K., Y. Zhu, E. Nissen, and Z. K. Shen (2021), On the Relevance of Geodetic Deformation Rates to Earthquake Potential, Geophys. Res. Lett., 48(11), doi:10.1029/2021gl093231.
