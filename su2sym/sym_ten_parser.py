import torch
import os

def parse_meta(s):
    f1=s[1:].split(", \'")
    meta=dict()
    for f in f1:
        sep_ind= f.find(":")
        meta[f[0:sep_ind-1]]=f[sep_ind+1:].strip()
    return meta

def parse_elem(s):
    # first occurance of "),"
    indt_end= s.find("),")
    inds= tuple([int(i) for i in s[1:indt_end].split(",")])
    val= eval(s[indt_end+2:-1])
    return inds, val
    
def parse_elems(s):
    raw_elems=[]
    level=0
    bf=""
    for c in s:
        bf+=c
        if c=="(":
            if level==0:
                bf=""
            level+=1
            continue
        if c==")":
            level-=1
            if level==0:
                raw_elems.append(bf)
                bf=""
            continue
    # parse tensor elems
    elems=[]
    for re in raw_elems:
        elems.append(parse_elem(re))
    return elems

def parse_classification(t):
    meta_t=[]
    raw_t=[]
    level=0
    meta=0
    elem=0
    bf=""
    for c in t:
        #print(c,end="")
        if level==0 and c=="(":
            level=1
            #print(f"{level} {meta} {elem}")
            continue

        if level==1 and c=="{":
            meta=1
            bf=""
            #print(f"{level} {meta} {elem}")
            continue

        if level*meta==1:
            if c=="}":
                meta=0
                #print(f"{level} {meta} {elem}")
                #print(bf)
                meta_t.append(bf)
                continue
            else:
                bf+=c
                #print("")
                continue

        if level*(1-meta)*(1-elem)==1 and c=="[":
            elem=1
            bf=""
            #print(f"{level} {meta} {elem}")
            continue

        if level*(1-meta)*elem==1:
            if c=="]":
                elem=0
                #print(f"{level} {meta} {elem}")
                raw_t.append(bf)
                #print(bf)
                continue
            else:
                bf+=c
                #print("")
                continue

        if level==1 and c==")":
            level=0
            continue

    meta=[{"meta": parse_meta(mt)} for mt in meta_t]    
    #for mt in meta_t:
    #     print(parse_meta(mt))
    elems=[parse_elems(rt) for rt in raw_t]
    # for elems in raw_t:
    #     print(parse_elems(elems))
    #     print(meta)
    #     print(elems)
    return [symten for symten in zip(meta,elems)]

def parse_symten_file(infile):
    with open(infile,"r") as f:
        data = f.read().replace('\n', '')
        return parse_classification(data)

def fill_from_sparse_coo_FIX(t,elems):
    """
    :param elems: non-zero elements defined in COO format (tuple(indices),value)
    :type elems: list[tuple(tuple(int),value)]
    """
    # for e in elems:
    #     t[e[0]]=e[1]

    elems[0]=(tuple([i+1 for i in elems[0][0]]), elems[0][1])
    for e in elems:
        t[ tuple([i-1 for i in e[0]]) ]=e[1]
    return t

def import_sym_tensors_FIX(p, D, pg, infile=None, dtype=torch.float64, device='cpu'):
    dims=(p,D,D,D,D)
    tensors=[]

    infile= f"{os.path.dirname(__file__)}/D{D}.txt" if infile is None else infile 

    tensors_coo= parse_symten_file(infile)
    for tcoo in tensors_coo:
        if pg==tcoo[0]["meta"]["pg"]:
            t= torch.zeros(dims, dtype=dtype, device=device)
            t= fill_from_sparse_coo_FIX(t, tcoo[1])
            tensors.append((tcoo[0],t))

    return tensors

def fill_from_sparse_coo(t,elems):
    """
    :param elems: non-zero elements defined in COO format (tuple(indices),value)
    :type elems: list[tuple(tuple(int),value)]
    """
    for e in elems:
        t[e[0]]=e[1]
    return t

def import_sym_tensors(p, D, pg, infile=None, dtype=torch.float64, device='cpu'):
    dims=(p,D,D,D,D)
    tensors=[]

    infile= f"{os.path.dirname(__file__)}/D{D}.txt" if infile is None else infile 

    tensors_coo= parse_symten_file(infile)
    print(tensors_coo)
    for tcoo in tensors_coo:
        if pg==tcoo[0]["meta"]["pg"]:
            t= torch.zeros(dims, dtype=dtype, device=device)
            t= fill_from_sparse_coo(t, tcoo[1])
            tensors.append((tcoo[0],t))

    return tensors