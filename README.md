# 3drem_based_uav_assisted_main
This code simulates a UAV-assisted heterogeneous network used for validating a 3D Radio Environment Map based positioning procedure for the UAV base station.

  - The simulation runs through and provides SNR and bitrates for every UE at a specific instance. Some UEs are served by the stationary ground base station, while others are associated to the drone base station. It is assumed throughout the simulation that at least one UE is served by the ground base station.
  
 - The simulator gives as outputs SINR, bitrate and energy efficiency. 

The code was run on Python 3.9. It requires installing the Path Loss modes from https://github.com/UPNAdrone/uav-radio. 

For more information, see the paper and the cited references. The paper was accepted and presented at 2025 IEEE GLOBECOM Workshops. It is not yet published but may be accessed on Zenodo. If you use this code, we kindly request that you reference the publication:
```bibtex
@misc{ivanov20263d,
  author       = {Ivanov, Antoni and
                  Тonchev, Кrasimir and
                  Vlahov, Atanas and
                  Poulkov, Vladimir and
                  Manolova, Agata},
  title        = {3D REM-based Positioning Procedure for UAV-
                   Assisted Het-Nets
                  },
  month        = feb,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18762008},
  url          = {https://doi.org/10.5281/zenodo.18762008},
}
```
