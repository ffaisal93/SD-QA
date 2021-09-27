import os
import pandas as pd
import pickle
from jiwer import wer
import pathlib

import fastwer

lang_st = {
    'ben':'bengali',
    'eng':'english',
    'ara':'arabic',
    'kor':'korean',
    'swa':'swahili'
}

asr_codes = {
    'ben':'bn',
    'eng':'en',
    'ara':'ar',
    'swa':'sw',
    'kor':'ko'
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
        
def a():
    pass

def make_dir(path):
    # define the name of the directory to be created

    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
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
   
def discardMissingWer(all_data,lang, min_or_all):
    if lang=='eng' and min_or_all==True:
        all_data = all_data[all_data['kenya-en-KE']!=""]
    if lang=='swa' and min_or_all==True:
        all_data = all_data[all_data['tanzania-sw-TZ']!=""]
    if lang=='kor' and min_or_all==True:
        all_data = all_data[all_data['kors-ko-KR']!=""]
    return all_data


def all_wer(root_path, f0, discard):
    # root_path = '../'
    # discard = ['.DS_Store','readme.md','asr_metadata']
    # needed=['test']
    min_or_all = True
    wer_info = []
    # for f0 in os.listdir(root_path):
    #     if f0 not in discard:
    for f1 in os.listdir(os.path.join(root_path,f0)):
        if f1 not in discard:
            print(f0,f1,'=================================')
            all_data  = pd.read_csv(os.path.join(root_path,f0,f1,'{}.{}.csv'.format(
                lang_st[f1],f0)), sep=",")
            del all_data['Unnamed: 0']
            all_data.fillna("",inplace=True)
            all_data[lang_st[f1]+'-gold'] = all_data.apply(lambda row: row[lang_st[f1]+'-gold'][:-1],axis=1)
            all_data[lang_st[f1]+'-gold'] = all_data.apply(lambda row: row[lang_st[f1]+'-gold'][:-1]
                                                            if row[lang_st[f1]+'-gold'][-1]==' '
                                                            else row[lang_st[f1]+'-gold'],axis=1)
            # if f1=='eng' and min_or_all==True:
            #     all_data = all_data[all_data['kenya-en-KE']!=""]
            all_data = discardMissingWer(all_data, f1, min_or_all)
            for col in list(all_data.columns)[3:]:
                region, asr_code = col.split('-')[0], col.split('-')[1]
                if asr_codes[f1]==asr_code:
                    ref = all_data[lang_st[f1]+'-gold'].values
                    main = all_data[col].values
#                             ref =  ref[main!=""]
#                             main = main[main!=""]
                    # print("col:{} main_c:{} ref_c:{}".format(col,len(main),len(ref)))
                    ref = list(map(str.lower,ref))
                    main = list(map(str.lower,main))
                    werror = fastwer.score(main, ref)
                    cerror = fastwer.score(main, ref, char_level=True)
                    # print(col, werror, cerror, asr_code)
                    wer_info.append([f1, region, col[-5:], round(werror,2), round(cerror,2),
                                        len(main), list(all_data[col]).count("")])
    cols  = ['lang','dialect','asr','wer','cer','count','asr-error-count']
    all_wer = pd.DataFrame.from_records(wer_info, columns=cols)
    return all_wer



def wer_gender(all_data, g_meta, gender, col, f2):
    dial = col.split('-')[0]
#     print(dial,col)
    dial_df_m = all_data[all_data['ID'].isin(g_meta[dial][gender])]
    if dial_df_m.shape[0]!=0:
        ref_m = dial_df_m[lang_st[f2]+'-gold'].str.lower().values
        hyp_m = dial_df_m[col].str.lower().values
        wer_v = round(wer(list(ref_m), list(hyp_m))*100,2)
        return wer_v, len(list(ref_m))
    else:
        return 'undefined', dial_df_m.shape[0]


def wer_gender_all(root_path,f1, discard):
    wer_df = {}
    min_or_all = True
    for f2 in os.listdir(os.path.join(root_path,f1)):
        if f2 not in discard:
            print(f2,"==============================")
            g_meta = {}
            lang_file = os.path.join(root_path,f1,f2,'{}.{}.csv'.format(lang_st[f2], f1))
            all_data = pd.read_csv(lang_file, sep=",")
            del all_data['Unnamed: 0']
            all_data.fillna("",inplace=True)
            all_data[lang_st[f2]+'-gold'] = all_data.apply(lambda row: row[lang_st[f2]+'-gold'][:-1],axis=1)
            all_data[lang_st[f2]+'-gold'] = all_data.apply(lambda row: row[lang_st[f2]+'-gold'][:-1]
                                                        if row[lang_st[f2]+'-gold'][-1]==' '
                                                        else row[lang_st[f2]+'-gold'],axis=1)
            all_data = discardMissingWer(all_data, f2, min_or_all)
            for f3 in os.listdir(os.path.join(root_path,f1,f2)):
                if f3 not in discard and dir_check(os.path.join(root_path,f1,f2,f3),'dir'):
                    meta_file = os.path.join(root_path,f1,f2,f3,'metadata.csv')
                    meta_df = pd.read_csv(meta_file, sep=",")
                    del meta_df['Unnamed: 0']
    #                 m_meta = meta_df.loc[meta_df['gender']=='male']
    #                 f_meta = meta_df.loc[meta_df['gender']=='female']
                    g_meta[f3] = {}
                    g_meta[f3]['male']=list(meta_df.loc[meta_df['gender']=='male', 'example_id'])
                    g_meta[f3]['female']=list(meta_df.loc[meta_df['gender']=='female', 'example_id'])
            for col in all_data.columns[3:]:
                g= 'male'
                wer_v = wer_gender(all_data, g_meta, g, col, f2)
                print(f2,col,g,wer_v)
                g= 'female'
                wer_v = wer_gender(all_data, g_meta, g, col, f2)
                print(f2,col,g,wer_v)

def wer_age(all_data, g_meta, gender, col, f2):
    dial = col.split('-')[0]
    dial_df_m = all_data[all_data['ID'].isin(g_meta[dial][gender])]
    if dial_df_m.shape[0]!=0:
        ref_m = dial_df_m[lang_st[f2]+'-gold'].str.lower().values
        hyp_m = dial_df_m[col].str.lower().values
        wer_v = round(wer(list(ref_m), list(hyp_m))*100,2)
        return wer_v, len(list(ref_m))
    else:
        return 'undefined', dial_df_m.shape[0]



def wer_age_all(root_path,f1, discard):
# f1='test'
    wer_df = {}
    min_or_all = True
    for f2 in os.listdir(os.path.join(root_path,f1)):
        if f2 not in discard:
            print(f2,"==============================")
            g_meta = {}
            lang_file = os.path.join(root_path,f1,f2,'{}.{}.csv'.format(lang_st[f2], f1))
            all_data = pd.read_csv(lang_file, sep=",")
            del all_data['Unnamed: 0']
            all_data.fillna("",inplace=True)
            all_data[lang_st[f2]+'-gold'] = all_data.apply(lambda row: row[lang_st[f2]+'-gold'][:-1],axis=1)
            all_data[lang_st[f2]+'-gold'] = all_data.apply(lambda row: row[lang_st[f2]+'-gold'][:-1]
                                                        if row[lang_st[f2]+'-gold'][-1]==' '
                                                        else row[lang_st[f2]+'-gold'],axis=1)
            all_data = discardMissingWer(all_data, f2, min_or_all)
            for f3 in os.listdir(os.path.join(root_path,f1,f2)):
                if f3 not in discard and dir_check(os.path.join(root_path,f1,f2,f3),'dir'):
                    meta_file = os.path.join(root_path,f1,f2,f3,'metadata.csv')
                    meta_df = pd.read_csv(meta_file, sep=",")
                    del meta_df['Unnamed: 0']
                    meta_df['age'] = meta_df.apply(lambda row: 0 if row['age'] in discard else row['age'],axis=1)
                    meta_df['age'] = meta_df.apply(lambda row: int(row['age']) if type(row['age'])==str 
                                                and row['age']!='na'
                                                else 0 if row['age']=='na'
                                                else row['age'],axis=1)
                    meta_df = meta_df[meta_df['age']!=0]
    #                 print(root_path,f1,f2,f3,set(meta_df['age'].values))
    #                 m_meta = meta_df.loc[meta_df['gender']=='male']
    #                 f_meta = meta_df.loc[meta_df['gender']=='female']
    #                 g_meta[f3] = dict(zip(meta_df['example_id'], meta_df['age']))
                    g_meta[f3]={}
                    g_meta[f3]['18-30']=list(meta_df.loc[meta_df['age']<31, 'example_id'])
                    g_meta[f3]['31-45']=list(meta_df.loc[(meta_df['age']>30) &
                                                        (meta_df['age']<=45), 'example_id'])
                    g_meta[f3]['46-']=list(meta_df.loc[(meta_df['age']>45), 'example_id'])
            for col in all_data.columns[3:]:
                g= '18-30'
                wer_v = wer_age(all_data, g_meta, g, col, f2)
                print(f2,col,g,wer_v)
                g= '31-45'
                wer_v = wer_age(all_data, g_meta, g, col, f2)
                print(f2,col,g,wer_v)
                g= '46-'
                wer_v = wer_age(all_data, g_meta, g, col, f2)
                print(f2,col,g,wer_v)
    #             dial_df_f = all_data[all_data['ID'].isin(g_meta[dial]['female'])]