import csv
import gzip
import json    
import os
import argparse
from log import logger
import multiprocessing
from functools import partial

def read_candidates_from_one_split(file_obj, lang, from_speech):
    """Read candidates from a single jsonl file."""
    candidates_list = []
    inserted = []
    i=0
    for line in file_obj:
        json_dict = json.loads(line)
        # if lang=='all':
        if i%100000==0:
            logger.info("{} examples done".format(i))
        #     candidates_list.append(json_dict)
        i+=1
        # elif json_dict['language']==lang:
        if str(json_dict['example_id']) in from_speech:
            json_dict['question_text'] = from_speech[str(json_dict['example_id'])]
            if str(json_dict['example_id']) not in inserted:
                candidates_list.append(json_dict)
                inserted.append(str(json_dict['example_id']))
#         candidates_dict[
#             json_dict["example_id"]] = json_dict["passage_answer_candidates"]
    logger.info("total example: {}".format(len(candidates_list)))
    return candidates_list



def read_reference(f, col, qmark=None):
    data = {}
    with open(f) as inp:
        csv_reader = csv.reader(inp, delimiter=',')
        firstline = True
        for l in csv_reader:
            # Skip the first line
            if firstline:
                firstline = False
                c = l.index(col)
                ind = l.index('ID')
                continue
            sid = l[int(ind)]
            sent = l[int(c)]
            if sent!='' and sent!=None:
                if qmark=='add' and sent[-2:]!=' ?' and sent[-1]!='?':
                    sent = sent + ' ?'
                elif qmark=='del' and sent[-2:]==' ?':
                    sent = sent[:-2]
                data[sid] = sent

    return data

def write_out(line):
#     logger.info("processing {}".format(line['example_id']))
    return json.dumps(line, ensure_ascii=False)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--datapath', type=str, help='data folder location.', required=True)
    parser.add_argument('-l', '--lang', type=str, help='Language.', required=True)
    parser.add_argument('-inpf', '--inpfile', type=str, help='Input data filename', required=True)
    parser.add_argument('-testp', '--testpath', type=str, help='Test data folder location.', required=True)
    parser.add_argument('-testf', '--testfile', type=str, help='Test data filename', required=True)
    parser.add_argument('-oupf', '--outfile', type=str, help='Output fliename.', required=True)
    parser.add_argument('-outp', '--outpath', type=str, help='Output folder location.', required=True)
    parser.add_argument('-tcol', '--testcol', type=str, help='test data column name.', required=True)
    parser.add_argument('-mcol', '--matchcol', type=str, help='reference column to match values.', required=False)
    parser.add_argument('-qmark', '--qmark', type=str, help='add/del question mark.', required=False)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = get_arguments()

    lang = args.lang
    f = args.inpfile
    datapath = args.datapath
    testpath = args.testpath
    filename = args.testfile
    col = args.testcol
    outfile = args.outfile
    outpath = args.outpath
    mcol = args.matchcol
    qmark  =  args.qmark    

    ##load speech transcript data to dictionary
    from_speech  = read_reference(os.path.join(testpath, filename),col, qmark)
    logger.info("total lines: {}".format(len(from_speech)))
    logger.info('column: {}'.format(col))
    if mcol!=None:
        logger.info('match column: {}'.format(mcol))
        if mcol=='self':
            diff_items = from_speech
        else:
            ref_text = read_reference(os.path.join(testpath, filename),mcol)
            logger.info("ref_text length: {}".format(len(ref_text)))
            if 'gold' in mcol:
                diff_items = {k: from_speech[k] for k in from_speech if k in ref_text
                 and from_speech[k] != ref_text[k] and from_speech[k]!=''}
            else:
                diff_items = {k: from_speech[k] for k in from_speech if ref_text[k]=='False'}
            
        from_speech = diff_items
        logger.info("total lines after discarding similars: {}".format(len(from_speech)))
    
    ##load original data into list of json dictionary and data replacement
    f_name, f_ext = os.path.splitext(f)
    if f_ext=='.gz':
        with gzip.open(os.path.join(datapath,f), "rb") as f:
            data = read_candidates_from_one_split(f, lang, from_speech)
    # ##data replacement
    # for line in data:
    #     if str(line['example_id']) in from_speech:
    #         line['question_text'] = from_speech[str(line['example_id'])]
    #save to file
    # dumps = []
    # with gzip.open(os.path.join(outpath,outfile), "w") as output_file:
    #     for prediction in data:
    #         if str(prediction['example_id']) in from_speech:
    #             dumps.append(json.dumps(prediction, ensure_ascii=False))
    #     logger.info("total lines while json dumping: {}".format(len(dumps)))
    #     output_file.write('\n'.join(dumps).encode())

    with gzip.open(os.path.join(outpath,outfile), "w") as output_file:
        p = multiprocessing.Pool(processes = multiprocessing.cpu_count()-1)
        lines  = partial(write_out)
        dumps = p.map(lines, data)
        p.close()
        p.join()
    # #         for prediction in data:
    # #             dumps.append(json.dumps(prediction, ensure_ascii=False))
        logger.info("total lines while json dumping: {}".format(len(dumps)))
        output_file.write('\n'.join(dumps).encode())