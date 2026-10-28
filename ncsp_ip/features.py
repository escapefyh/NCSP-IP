"""TRAIN-only name features and the original 92-dimensional annotations."""
import re
from itertools import product
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

KINDS = [
    "fragment_candidate",
    "variant_candidate",
    "specific_trna_candidate",
    "broad_or_unclear",
    "other_gene_candidate",
]

AMINO_ACIDS = [
    "Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu",
    "Gly", "His", "Ile", "Leu", "Lys", "Met", "Phe",
    "Pro", "Ser", "Thr", "Trp", "Tyr", "Val", "Sec",
]

TRIPLETS = ["".join(x) for x in product("ACGT", repeat=3)]

def name_info(raw_name):
    # 只生成检索提示，绝不修改节点的原始名称。
    name = raw_name.replace("\u00a0", " ").strip()
    low = name.lower()
    notes = []

    if re.search(r"(^|[-_\s])(?:mt-)?nd[1-6]", low):
        kind = "other_gene_candidate"
        notes.append("Possible protein-coding mitochondrial gene")
    elif re.search(
        r"trf|tirna|tsrna|tdr|half-|misc_|^ts-|^5p_", low
    ):
        kind = "fragment_candidate"
    elif re.search(
        r"\d+\s*[ACGT]\s*>\s*[ACGT]|[ACGT]\d+[ACGT]", name, re.I
    ):
        kind = "variant_candidate"
    elif re.fullmatch(
        r"TR[A-Z]-[ACGT]{3}\d+-\d+(?:\.\d+)?", name, re.I
    ) or re.fullmatch(
        r"tRNA-[A-Za-z]+-[ACGTU]{3}-\d+-\d+", name, re.I
    ):
        kind = "specific_trna_candidate"
    else:
        kind = "broad_or_unclear"

    if "|" in name or re.search(r"-\d+\s+[I|]$", name):
        notes.append("Possible extra trailing character")
    if re.search(r"G1u|GIn|(?:^|-)lle|VaI", name):
        notes.append("Possible letter/digit transcription issue")
    if re.search(r"[-_][oO]\d+", name):
        notes.append("Possible letter O versus digit zero")
    if re.fullmatch(r"tRF-\d+-[A-Z0-9]+", name, re.I):
        notes.append("Try MINTplates decoding")

    aa = "unknown"
    for candidate in sorted(AMINO_ACIDS, key=len, reverse=True):
        pattern = (
            r"(?<![A-Za-z])" + candidate + r"(?=$|[^A-Za-z])"
        )
        prefixed = (
            r"(?:trna|trn|trf|tirna|tsrna)\s*-?"
            + candidate + r"(?=$|[^A-Za-z])"
        )
        if re.search(pattern, name, re.I) or re.search(
            prefixed, name, re.I
        ):
            aa = candidate
            break

    anticodon = "unknown"
    match = re.search(
        r"(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|"
        r"iMet|Met|Leu|Lys|Phe|Pro|Ser|Thr|Trp|Tyr|Val|Sec)"
        r"[\s_-]*([ACGTU]{3})(?=$|[^A-Za-z])",
        name,
        re.I,
    )
    if match:
        # 名称中的 DNA/RNA 字母统一为 T；不代表已经核验。
        anticodon = match.group(1).upper().replace("U", "T")

    return name, kind, aa, anticodon, "; ".join(notes)

def annotation(name):
 _,k,a,c,_=name_info(name);x=np.zeros(92);x[KINDS.index(k)]=1;x[5+(AMINO_ACIDS+['unknown']).index(a)]=1;x[27+(TRIPLETS+['unknown']).index(c)]=1;return x

def vectorize(names,A):
    ids=np.flatnonzero(A.sum(1)>0)
    v=TfidfVectorizer(analyzer='char',ngram_range=(2,4),max_features=2048,lowercase=False,dtype=np.float32)
    v.fit([names[i] for i in ids]);return v.transform(names).toarray(),v,ids
