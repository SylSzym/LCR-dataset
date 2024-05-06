import pandas as pd
import warnings

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

class AminoAcid:
    def __init__(self, name, synonyms):
        self.name = name
        self.synonyms = synonyms

glycine = AminoAcid('glycine', [' polyglycine ', ' poly-glycine ', ' glycine-rich ', ' polyG ', ' poly-G ', ' G-rich ', ' G-tract ', ' poly(G) ', ' poly(glicyne) ', ' poly-Gly '])
alanine = AminoAcid('alanine', [' polyalanine ', ' poly-alanine ', ' alanine-rich ', ' polyA ', ' poly-A ', ' A-rich ', ' A-tract ', ' poly(A) ', ' poly(alanine) ', ' poly-Ala '])
valine = AminoAcid('valine', [' polyvaline ', ' poly-valine ', ' valine-rich ', ' polyV ', ' poly-V ', ' V-rich ', ' V-tract ', ' poly(V) ', ' poly(valine) ', ' poly-Val '])
leucine = AminoAcid('leucine', [' polyleucine ', ' poly-leucine ', ' leucine-rich ', ' polyL ', ' poly-L ', ' L-rich ', ' L-tract ', ' poly(L) ', ' poly(leucine) ', ' poly-Leu '])
isoleucine = AminoAcid('isoleucine', ['polyisoleucine', 'poly-isoleucine', 'isoleucine-rich', 'polyI', 'poly-I', 'I-rich', 'I-tract', 'poly(I)', 'poly(isoleucine)', 'poly-Iso'])
serine = AminoAcid('serine', ['polyserine', 'poly-serine', 'serine-rich', 'polyS', 'poly-S', 'S-rich', 'S-tract', 'poly(S)', 'poly(serine)', 'poly-Ser'])
threonine = AminoAcid('threonine', ['polythreonine', 'poly-threonine', 'threonine-rich', 'polyT', 'poly-T', 'T-rich', 'T-tract', 'poly(T)', 'poly(threonine)', 'poly-Thr'])
cysteine = AminoAcid('cysteine', ['polycysteine', 'poly-cysteine', 'cysteine-rich', 'polyC', 'poly-C', 'C-rich', 'C-tract', 'poly(C)', 'poly(cysteine)', 'poly-Cys'])
methionine = AminoAcid('methionine', ['polymethionine', 'poly-methionine', 'methionine-rich', 'polyM', 'poly-M', 'M-rich', 'M-tract', 'poly(M)', 'poly(methionine)', 'poly-Met'])
phenylalanine = AminoAcid('phenylalanine', ['polyphenylalanine', 'poly-phenylalanine', 'phenylalanine-rich', 'polyF', 'poly-F', 'F-rich', 'F-tract', 'poly(F)', 'poly(phenylalanine)', 'poly-Phe'])
tyrosine = AminoAcid('tyrosine', ['polytyrosine', 'poly-tyrosine', 'tyrosine-rich', 'polyY', 'poly-Y', 'Y-rich', 'Y-tract', 'poly(Y)', 'poly(tyrosine)', 'poly-Tyr'])
tryptophan = AminoAcid('tryptophan', ['polytryptophan', 'poly-tryptophan', 'tryptophan-rich', 'polyW', 'poly-W', 'W-rich', 'W-tract', 'poly(W)', 'poly(tryptophan)', 'poly-Trp'])
aspartic_acid = AminoAcid('aspartic_acid', ['polyaspartic', 'poly-polyaspartic', 'polyaspartic-rich', 'polyD', 'poly-D', 'D-rich', 'D-tract', 'poly(D)', 'poly(aspartic)', 'poly-Asp'])
asparagine = AminoAcid('asparagine', ['polyasparagine', 'poly-asparagine', 'asparagine-rich', 'polyN', 'poly-N', 'N-rich', 'N-tract', 'poly(N)', 'poly(asparagine)', 'poly-Asn'])
glutamic_acid = AminoAcid('glutamic_acid', ['polyglutamic', 'poly-glutamic', 'glutamic-rich', 'polyE', 'poly-E', 'E-rich', 'E-tract', 'poly(E)', 'poly(glutamic)', 'poly-Gln'])
glutamine = AminoAcid('glutamine', ['polyglutamine', 'poly-glutamine', 'glutamine-rich', 'polyQ', 'poly-Q', 'Q-rich', 'Q-tract', 'poly(Q)', 'poly(glutamine)', 'poly-Glu'])
lysine = AminoAcid('lysine', ['polylysine', 'poly-lysine', 'lysine-rich', 'polyK', 'poly-K', 'K-rich', 'K-tract', 'poly(K)', 'poly(lysine)', 'poly-Lys'])
arginine = AminoAcid('arginine', ['polyarginine', 'poly-arginine', 'arginine-rich', 'polyR', 'poly-R', 'R-rich', 'R-tract', 'poly(R)', 'poly(arginine)', 'poly-Arg'])
histidine = AminoAcid('histidine', ['polyhistidine', 'poly-histidine', 'histidine-rich', 'polyH', 'poly-H', 'H-rich', 'H-tract', 'poly(H)', 'poly(histidine)', 'poly-His'])
proline = AminoAcid('proline', ['polyproline', 'poly-proline', 'proline-rich', 'polyP', 'poly-P', 'P-rich', 'P-tract', 'poly(P)', 'poly(proline)', 'poly-Pro'])


amino_list = [glycine,alanine,valine,leucine,isoleucine,serine,threonine,cysteine,methionine,phenylalanine,tyrosine,tryptophan,aspartic_acid,asparagine,glutamic_acid,glutamine,lysine,arginine,histidine,proline]

class Function:
    def __init__(self, name, synonyms):
        self.name = name
        self.synonyms = synonyms

antigen = Function('antigen', ['antigen'])
helix = Function('helix', ['helix'])
nucleic_acid_binding = Function('nucleic acid binding', ['DNA-binding', 'DNA binding', 'to binds DNA', 'RNA-binding', 'RNA binding', 'to binds RNA'])
kinase_activity = Function('kinase activity', ['kinase activity'])
phosphatase_activity = Function('phosphatase activity', ['phosphatase activity'])
transcription_factor = Function('transcription factor', ['transcription factor'])
zinc_finger = Function('zinc finger', ['zinc finger', 'zinc-finger'])

function_list = [antigen, helix, nucleic_acid_binding, kinase_activity, phosphatase_activity, transcription_factor, zinc_finger]

hard_substring = ['poly-L-', 'polyacrylamide', 'poly(A)-mRNA', 'acid-rich', 'poly-ADP-ribose', 'poly(amido amine)', 'poly-ADP-ribosylation', 'poly-L-ornithine', 'poly(A)(+) RNA', 'poly(A)-containing mRNA']

def check_amino_acid(title, abstract, amino):
    return any(substring in title for substring in amino.synonyms) or any(substring in abstract for substring in amino.synonyms)

def check_function(title, abstract, function):
    return any(fsubstring in title for fsubstring in function.synonyms) or any(fsubstring in abstract for fsubstring in function.synonyms)

def process_file(filename):
    data = pd.read_csv(filename, sep=';', dtype=object)
    data['LCR'] = '-'
    data['Function'] = '-'
    data['Hard'] = '-'
    data['Category'] = '-'

    for i in range(len(data)):

        for amino in amino_list:
            if check_amino_acid(data['Title'].iloc[i], data['Abstract'].iloc[i], amino):
                data['LCR'].iloc[i] = amino.name

        for function in function_list:
            if check_function(data['Title'].iloc[i], data['Abstract'].iloc[i], function):
                data['Function'].iloc[i] = function.name

        for hard in hard_substring:
            if (hard in str(data['Title'].iloc[i]) or (hard in str(data['Abstract'].iloc[i]))):
                data['Hard'].iloc[i] = hard

        if data['LCR'].iloc[i] != '-' and data['Function'].iloc[i] != '-' and data['Hard'].iloc[i] == '-':
            data['Category'].iloc[i] = 'LCR_with_function'
        if data['LCR'].iloc[i] != '-' and data['Function'].iloc[i] == '-' and data['Hard'].iloc[i] == '-':
            data['Category'].iloc[i] = 'LCR_without_function'
        if data['LCR'].iloc[i] != '-' and data['Function'].iloc[i] == '-' and data['Hard'].iloc[i] != '-':
            data['Category'].iloc[i] = 'LCR_with_hard'
        if data['LCR'].iloc[i] == '-' and data['Function'].iloc[i] == '-' and data['Hard'].iloc[i] == '-':
            data['Category'].iloc[i] = 'hard_0'
        if data['LCR'].iloc[i] == '-' and data['Function'].iloc[i] == '-' and data['Hard'].iloc[i] != '-':
            data['Category'].iloc[i] = 'hard_1'
        if data['LCR'].iloc[i] == '-' and data['Function'].iloc[i] != '-' and data['Hard'].iloc[i] == '-':
            data['Category'].iloc[i] = 'hard_2'
        if data['LCR'].iloc[i] == '-' and data['Function'].iloc[i] != '-' and data['Hard'].iloc[i] != '-':
            data['Category'].iloc[i] = 'hard_3'

    data.to_csv('./sample_data/annotated_' + filename.split('/')[3], sep=';', index=False)

def main():
    process_file('./sample_data/raw_data/sample_data.csv')

if __name__ == "__main__":
    main()
