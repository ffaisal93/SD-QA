import os
import pandas as pd
import pickle
from jiwer import wer

lang_st = {
    'ben':'bengali',
    'eng':'english',
    'ara':'arabic',
    'kor':'korean',
    'swa':'swahili'
}

def dir_check(path,types):
    ans = False
    if types=='dir':
        ans = os.path.isdir(path)
    elif types=='file':
        if os.path.isdir(path)==False:
            ans = os.path.exists(path)
        else:
            ans=False
    return ans
        

def make_dir(path):
    # define the name of the directory to be created

    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)


def calc_wer(root_path, f1, discard):
    wer_df = {}
    for f2 in os.listdir(os.path.join(root_path,f1)):
        if f2 not in discard:
            print(lang_st[f2])
            all_data  = pd.read_csv(os.path.join(root_path,f1,f2,'{}.{}.csv'.format(lang_st[f2],f1)), sep=",")
            del all_data['Unnamed: 0']
            all_data.fillna("",inplace=True)
            all_data[lang_st[f2]+'-gold'] = all_data.apply(lambda row: row[lang_st[f2]+'-gold'][:-1],axis=1)
            all_data[lang_st[f2]+'-gold'] = all_data.apply(lambda row: row[lang_st[f2]+'-gold'][:-1]
                                                           if row[lang_st[f2]+'-gold'][-1]==' '
                                                           else row[lang_st[f2]+'-gold'],axis=1)
            print(all_data.shape)
            wer_df[f2] =  pd.DataFrame()
            wer_df[f2]['ID'] = all_data['ID']
            gold = all_data.columns[2]
            for col in all_data.columns[3:]:
                col_name = '{}'.format(col)
                wer_df[f2][col_name] = all_data.apply(lambda row: round(wer(row[gold].lower(), row[col].lower()),3)
                                                  if row[col]!="" else 1, axis=1)
                wer_df[f2].to_csv(os.path.join(root_path,f1,f2,'wer.{}.{}.csv'.format(lang_st[f2], f1)))
    return wer_df


def get_metadata(root_path, f1, discard):
    meta_info = {}
    all_info = {}
    for f2 in os.listdir(os.path.join(root_path,f1)):
        if f2 not in discard:
            meta_info[f2]={}
            g_meta = {}
            lang_file = os.path.join(root_path,f1,f2,'{}.{}.csv'.format(lang_st[f2], f1))
            all_data = pd.read_csv(lang_file, sep=",")
            del all_data['Unnamed: 0']
            all_data.fillna("#NOTFOUND#",inplace=True)
            ###drop ? mark as no ? in transcript output
            all_data[lang_st[f2]+'-gold'] = all_data.apply(lambda row: row[lang_st[f2]+'-gold'][:-1],axis=1)
            all_data[lang_st[f2]+'-gold'] = all_data.apply(lambda row: row[lang_st[f2]+'-gold'][:-1]
                                                        if row[lang_st[f2]+'-gold'][-1]==' '
                                                        else row[lang_st[f2]+'-gold'],axis=1)
            if f2=='eng':
                all_data = all_data[all_data['kenya-en-KE']!="#NOTFOUND#"]
            all_info[f2] = all_data
            for f3 in os.listdir(os.path.join(root_path,f1,f2)):
                if f3 not in discard and dir_check(os.path.join(root_path,f1,f2,f3),'dir'):
                    meta_file = os.path.join(root_path,f1,f2,f3,'metadata.csv')
                    meta_df = pd.read_csv(meta_file, sep=",")
                    del meta_df['Unnamed: 0']
                    meta_info[f2][f3]= meta_df
                    if f2=='swa':
                        meta_info[f2][f3]['language']=meta_info[f2][f3].apply(lambda x: 
                                                                                        'swa' 
                                                                                        if x['language']=='swh'
                                                                                        else 'swa'
                                                                                        ,axis=1)
                        meta_info[f2][f3].to_csv(meta_file)
    return meta_info, all_info
