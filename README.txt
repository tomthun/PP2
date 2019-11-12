NES/NLS Data Set

Files:
- nes_nls.fasta
- nes_nls.tab


The FASTA file contains the protein sequences and their respective IDs in the header (e.g. 'Q09472').

The TAB file contains the annotations regarding NES/NLS.
Each line contains a protein ID, the start and stop position (both inclusive) of a NES/NLS and the type (if NES or NLS).
Note 1: Indices start at 1, not 0. Thus, "1" refers to the first residue in the sequence.
Note 2: A single protein can have multiple NES and/or NLS.



Prediction Task: Multi-Class classification (3 classes)
- Predict if a residue is part of a NES, NLS or neither.
