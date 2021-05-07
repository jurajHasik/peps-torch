import argparse 

parser = argparse.ArgumentParser(description='Read results of examples/ctmrg_j1j1_c4v.py.')
parser.add_argument('f', metavar='file', type=str, help='name of the file')
parser
args = parser.parse_args()


def searcher(inf, string):
    with open("SpecRes.txt", 'a') as fo:
        with open(inf, "r") as fi:
            for line in fi:
                if string in line:
                    fo.write(f"\n{inf}\n")
                    for line in fi:
                        fo.write(line)

if __name__ == '__main__':
    searcher(inf=args.f, string="spectrum(T)")