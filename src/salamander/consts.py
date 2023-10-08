NUCLEOTIDES = ["A", "C", "G", "T"]

SBS_TYPES_6 = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
SBS_TYPES_96 = [
    f"{n1}[{sbs_6}]{n2}"
    for sbs_6 in SBS_TYPES_6
    for n1 in NUCLEOTIDES
    for n2 in NUCLEOTIDES
]

# fmt: off
INDEL_TYPES_83 = [
    "DEL.C.1.1", "DEL.C.1.2", 'DEL.C.1.3', "DEL.C.1.4", "DEL.C.1.5", "DEL.C.1.6+",
    "DEL.T.1.1", "DEL.T.1.2", 'DEL.T.1.3', "DEL.T.1.4", "DEL.T.1.5", "DEL.T.1.6+",
    "INS.C.1.0", "INS.C.1.1", 'INS.C.1.2', "INS.C.1.3", "INS.C.1.4", "INS.C.1.5+",
    "INS.T.1.0", "INS.T.1.1", 'INS.T.1.2', "INS.T.1.3", "INS.T.1.4", "INS.T.1.5+",
    "DEL.repeats.2.1", "DEL.repeats.2.2", "DEL.repeats.2.3",
    "DEL.repeats.2.4", "DEL.repeats.2.5", "DEL.repeats.2.6+",
    "DEL.repeats.3.1", "DEL.repeats.3.2", "DEL.repeats.3.3",
    "DEL.repeats.3.4", "DEL.repeats.3.5", "DEL.repeats.3.6+",
    "DEL.repeats.4.1", "DEL.repeats.4.2", "DEL.repeats.4.3",
    "DEL.repeats.4.4", "DEL.repeats.4.5", "DEL.repeats.4.6+",
    "DEL.repeats.5+.1", "DEL.repeats.5+.2", "DEL.repeats.5+.3",
    "DEL.repeats.5+.4", "DEL.repeats.5+.5", "DEL.repeats.5+.6+",
    "INS.repeats.2.0", "INS.repeats.2.1", "INS.repeats.2.2",
    "INS.repeats.2.3", "INS.repeats.2.4", "INS.repeats.2.5+",
    "INS.repeats.3.0", "INS.repeats.3.1", "INS.repeats.3.2",
    "INS.repeats.3.3", "INS.repeats.3.4", "INS.repeats.3.5+",
    "INS.repeats.4.0", "INS.repeats.4.1", "INS.repeats.4.2",
    "INS.repeats.4.3", "INS.repeats.4.4", "INS.repeats.4.5+",
    "INS.repeats.5+.0", "INS.repeats.5+.1", "INS.repeats.5+.2",
    "INS.repeats.5+.3", "INS.repeats.5+.4", "INS.repeats.5+.5+",
    "DEL.MH.2.1",
    "DEL.MH.3.1", "DEL.MH.3.2",
    "DEL.MH.4.1", "DEL.MH.4.2", "DEL.MH.4.3",
    "DEL.MH.5+.1", "DEL.MH.5+.2", "DEL.MH.5+.3", "DEL.MH.5+.4", "DEL.MH.5+.5+"
]
# fmt: on

# 10 colors
COLORS_MATHEMATICA = [
    (0.368417, 0.506779, 0.709798),
    (0.880722, 0.611041, 0.142051),
    (0.560181, 0.691569, 0.194885),
    (0.922526, 0.385626, 0.209179),
    (0.528288, 0.470624, 0.701351),
    (0.772079, 0.431554, 0.102387),
    (0.363898, 0.618501, 0.782349),
    (1.0, 0.75, 0.0),
    (0.280264, 0.715, 0.429209),
    (0.0, 0.0, 0.0),
]

# Trinucleotide colors for the 96 dimensional mutation spectrum
COLORS_TRINUCLEOTIDES = [
    (0.33, 0.75, 0.98),
    (0.0, 0.0, 0.0),
    (0.85, 0.25, 0.22),
    (0.78, 0.78, 0.78),
    (0.51, 0.79, 0.24),
    (0.89, 0.67, 0.72),
]

COLORS_SBS96 = [COLORS_TRINUCLEOTIDES[i // 16] for i in range(96)]

COLORS_INDEL = [
    "#FCBD6F",  # 1bp Del C
    "#FD8001",  # 1bp Del T
    "#B0DC8B",  # 1bp Ins C
    "#35A02E",  # 1bp Ins T
    "#FCC9B4",  # 2bp Del Repeats
    "#FC896B",  # 3bp Del Repeats
    "#F04432",  # 4bp Del Repeats
    "#BC1A1A",  # 5+ bp Del Repeats
    "#CFE0F0",  # 2bp Ins Repeats
    "#94C3DF",  # 3bp Ins Repeats
    "#4A98C8",  # 4bp Ins Repeats
    "#1665AA",  # 5+ bp Ins Repeats
    "#E1E0ED",  # 2bp Del MH
    "#B5B5D8",  # 3bp Del MH
    "#8683BC",  # 4bp Del MH
    "#624099",  # 5+bp Del MH
]

# 12 * 6 + 11 = 83 colors
n_times = 12 * [6] + [1, 2, 3, 5]
COLORS_INDEL83 = [n * [col] for n, col in zip(n_times, COLORS_INDEL)]
COLORS_INDEL83 = [col for color_list in COLORS_INDEL83 for col in color_list]
