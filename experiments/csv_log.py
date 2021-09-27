import os
import pandas as pd
import pickle
import argparse



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='+', default=[], required=True)
    parser.add_argument('-c', '--cols', nargs='+', default=[], required=True)
    parser.add_argument('-d', '--dir', type=str, help='Directory path', required=True)
    parser.add_argument('-o', '--out', type=str, help='Output filename', required=True)
    args = parser.parse_args()
    return args

    

if __name__ =='__main__':
    args = get_arguments()
    files = args.files
    cols = args.cols
    dirname = args.dir
    outfile = args.out
    df = pd.DataFrame()
    all_id=[]
    data_dict={}
    for i,filename in enumerate(files):
        with open(os.path.join(dirname,filename), 'rb') as f:
            l = pickle.load(f)
        if i==0:
            for xx in l:
                ids = int(xx.split('\n')[0][1:-1])
                data_dict[ids]={cols[i]:'\n'.join(xx.split('\n')[1:])}
            # df[cols[i]]=l
        elif i>0:
            for xx in l:
                ids = int(xx.split('\n')[0][1:-1])
                data_dict[ids].update({cols[i]:'\n'.join(xx.split('\n')[1:])})
    # print(data_dict)
    df = pd.DataFrame.from_dict(data_dict).T
    df.reset_index(inplace=True)
    df.to_csv(os.path.join(outfile))