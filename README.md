# 3drem_based_uav_assisted_main
This code simulates a UAV-assisted heterogeneous network used for validating a 3D Radio Environment Map based positioning procedure for the UAV base station (2025 GLOBECOM Workshops paper entitled "3D REM-based Positioning Procedure for UAV-Assisted Het-Nets").

  - The simulation runs through and provides SNR and bitrates for every UE at a specific instance. Some UEs are served by the stationary ground base station, while others are associated to the drone base station. It is assumed throughout the simulation that at least one UE is served by the ground base station.
  
 - The simulator gives as outputs SINR, bitrate and energy efficiency. 

The code was run on Python 3.9. It requires installing the Path Loss modes from https://github.com/UPNAdrone/uav-radio. 

For more information, see the paper and the cited references. The paper was accepted and presented at 2025 IEEE GLOBECOM Workshops. It is not yet published but may be accessed on Zenodo. If you use this code, we kindly request that you reference the publication:
```bibtex
@INPROCEEDINGS{11591168,
  author={Ivanov, Antoni and Tonchev, Krasimir and Vlahov, Atanas and Poulkov, Vladimir and Manolova, Agata},
  booktitle={2025 IEEE Globecom Workshops (GC Wkshps)}, 
  title={3D REM-based Positioning Procedure for UAV-Assisted Het-Nets}, 
  year={2025},
  volume={},
  number={},
  pages={2320-2326},
  doi={10.1109/GCWkshps68340.2025.11591168}
}
```
