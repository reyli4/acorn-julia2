This directory holds the NYS grid network information. These files are the result of the network reduction in [Liu et al. (2023)](https://ieeexplore.ieee.org/document/9866561).

### Bus information
- `bus_prop_boyuan.csv`: Bus properties from [Bo's python repo](https://github.com/boyuan276/NYgrid-python)
- `bus_prop_liu_etal_2024.csv`: Bus properties from the [Liu et al. (2024) repo](https://github.com/AndersonEnergyLab-Cornell/ny-clcpa2050)
- Note that these files are identical other than column 3 (real power demand). They also agree with Elnaz's 2030 repo.
- *Additional bus information* is given in `npcc_new.csv`, taken from [Bo's python repo](https://github.com/boyuan276/NYgrid-python). This seems to have been amended from the original NPCC 140-bus system information in [PSAT](http://faraday1.ucd.ie/psat.html). I could not verify this other than finding the original csv file [here](https://github.com/CURENT/andes/tree/master/andes/cases).
- As far as I can tell, only the bus network connections and lat/lon locations are used. The other information does not enter into the OPF analysis.

Bus Data Format:
| Column | Description | Notes |
|--------|-------------|-------|
| 1 | Bus number | Positive integer |
| 2 | Bus type | 1 = PQ bus, 2 = PV bus, 3 = Reference bus, 4 = Isolated bus |
| 3 | Pd | Real power demand (MW) |
| 4 | Qd | Reactive power demand (MVAr) |
| 5 | Gs | Shunt conductance (MW demanded at V = 1.0 p.u.) |
| 6 | Bs | Shunt susceptance (MVAr injected at V = 1.0 p.u.) |
| 7 | Area number | Positive integer |
| 8 | Vm | Voltage magnitude (p.u.) |
| 9 | Va | Voltage angle (degrees) |
| 10 | baseKV | Base voltage (kV) |
| 11 | Zone | Loss zone (positive integer) |
| 12 | maxVm | Maximum voltage magnitude (p.u.) |
| 13 | minVm | Minimum voltage magnitude (p.u.) |

### Line information
- `branch_prop_boyuan.csv`: Branch properties from [Bo's python repo](https://github.com/boyuan276/NYgrid-python)
- `branch_prop_liu_etal_2024.csv`: Branch properties from the [Liu et al. (2024) repo](https://github.com/AndersonEnergyLab-Cornell/ny-clcpa2050)
- These files are mostly identical. There are 3 additional branches in Bo's, representing 2 external and 1 internal connections. The flow limits are also different for some lines.
- Note that both contain a duplicate (connecting buses 39-73). Kabir 2024 (2030 grid) agrees with Liu et al 2024.
- Original data taken from NPCC 140-bus system as verified [here](https://github.com/CURENT/andes/tree/master/andes/cases), plus some additional data related to flow limits. 

Branch Data Format:
| Column | Description | Notes |
|--------|-------------|-------|
| 1 | From bus number | Positive integer |
| 2 | To bus number | Positive integer |
| 3 | r | Resistance (p.u.) |
| 4 | x | Reactance (p.u.) |
| 5 | b | Total line charging susceptance (p.u.) |
| 6 | rateA | MVA rating A (long term rating) |
| 7 | rateB | MVA rating B (short term rating) |
| 8 | rateC | MVA rating C (emergency rating) |
| 9 | ratio | Transformer off nominal turns ratio (= 0 for lines). Taps at 'from' bus, impedance at 'to' bus. If r = x = 0, then ratio = Vf / Vt |
| 10 | angle | Transformer phase shift angle (degrees), positive => delay |
| 11 | status | Initial branch status, 1 - in service, 0 - out of service |
| 12 | ANGMIN | Minimum angle difference, angle(Vf) - angle(Vt) (degrees) |
| 13 | ANGMAX | Maximum angle difference, angle(Vf) - angle(Vt) (degrees). Angle difference is unbounded below if ANGMIN < -360 and above if ANGMAX > 360. If both are zero, it is unconstrained. |

### Generator information

- `gen_prop_boyuan.csv`: Generator matrix from [Bo's python repo](https://github.com/boyuan276/NYgrid-python)
- `gen_prop_liu_etal_2024.csv`: Generator matrix from the [Liu et al. (2024) repo](https://github.com/AndersonEnergyLab-Cornell/ny-clcpa2050)
- This data seems to come from a variety of sources, as described in the 2019 paper. A basic list of generator names and lats/lons is given on the [NYISO website](http://mis.nyiso.com/public/) but this is appended with additional information.
- These two csv files don't really agree -- some generators can be matched across the files (mainly hydro and nuclear) but there are differences in generation parameters. The import "generators" do not match.
- `gencost_prop_boyuan.csv`: Generator cost information [Bo's python repo](https://github.com/boyuan276/NYgrid-python). 

Generator Data Format:
| Column | Field | Description | Notes |
|--------|-------|-------------|--------|
| 1 | bus number | Bus number | Positive integer |
| 2 | Pg | Real power output | MW |
| 3 | Qg | Reactive power output | MVAr |
| 4 | Qmax | Maximum reactive power output | MVAr |
| 5 | Qmin | Minimum reactive power output | MVAr |
| 6 | Vg | Voltage magnitude setpoint | p.u. |
| 7 | mBase | Total MVA base of machine | Defaults to baseMVA |
| 8 | status | Machine service status | > 0: in service, ≤ 0: out of service |
| 9 | Pmax | Maximum real power output | MW |
| 10 | Pmin | Minimum real power output | MW |
| 11 | Pc1 | Lower real power output of PQ capability curve | MW |
| 12 | Pc2 | Upper real power output of PQ capability curve | MW |
| 13 | Qc1min | Minimum reactive power output at Pc1 | MVAr |
| 14 | Qc1max | Maximum reactive power output at Pc1 | MVAr |
| 15 | Qc2min | Minimum reactive power output at Pc2 | MVAr |
| 16 | Qc2max | Maximum reactive power output at Pc2 | MVAr |
| 17 | ramp rate | Load following/AGC ramp rate | MW/min |
| 18 | ramp rate | 10 minute reserves ramp rate | MW |
| 19 | ramp rate | 30 minute reserves ramp rate | MW |
| 20 | ramp rate | Reactive power ramp rate (2 sec timescale) | MVAr/min |
| 21 | APF | Area participation factor | - |