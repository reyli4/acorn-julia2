Interface limits are taken from NYISO Reliability Needs Assessment (RNA). All units are MW.

[2020 RNA](https://www.nyiso.com/documents/20142/2248793/2020-RNAReport-Nov2020.pdf)
- Vivienne's paper relied on 2020 RNA Figure 48
- `if_lim_2020_rna.csv` is my parsing of the RNA
- `vivienne_2023_paper.csv` matches Vivenne's table

[2022 RNA](https://www.nyiso.com/documents/20142/2248793/2022-RNA-Report.pdf)
- Figure 34 provides updated limits

[2024 RNA (draft)](https://www.nyiso.com/documents/20142/47773760/2024RNA_Report_103124MC.pdf/956d57b8-0a30-d1fb-70a1-9e1680ecdb6f)
- Interface limits from Slide 41 [here](https://www.nyiso.com/documents/20142/46031967/2024RNA_PrelimResults_July25ESPWG-TPAS.pdf/f635a8ab-458f-35e5-ef92-0847c0ea6bca)
- `if_lim_2024_rna.csv` is my parsing of the RNA

Other sources:
- `boyuan.csv` is taken from [Bo's python repo[(https://github.com/boyuan276/NYgrid-python)

Notes:
- `if_lim_map.csv` is the mapping of bus number to interface limit. A negative bus number indicates negative flow across the interface, and vice-versa. This mapping is taken from Vivienne's model (Feb 2025 -- should double check this). 