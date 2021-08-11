import codecs
import os
from collections import defaultdict
import random
import numpy as np
import pandas

def read_all_lemmas(path):
    """
    Return a sorted list of all the lemmas in the file. No need to save the forms and features!
    """
    data = open(path, encoding='utf8').read().split('\n')
    lemmas = [line.split('\t')[0] for line in data if not line in {'', ' '}]
    lemmas.sort()
    return lemmas

def check_lemma_split(train_path, dev_path, test_path):
    """
    Apply read_all_lemmas 3 times to get the lists of all the lemmas. Then, observe the lists' intersections with each other. If they aren't empty, return False!!!
    """
    train_lemmas, dev_lemmas, test_lemmas = read_all_lemmas(train_path), read_all_lemmas(dev_path), read_all_lemmas(test_path)
    lang = os.path.splitext(test_path)[0][-3:]
    inter1 = set(train_lemmas) & set(dev_lemmas)
    inter2 = set(train_lemmas) & set(test_lemmas)
    inter3 = set(dev_lemmas) & set(test_lemmas)
    return bool(inter1), bool(inter2), bool(inter3)


def read(fname):
    """ read file name """
    D = {}
    with codecs.open(fname, 'rb', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lemma, word, tag = line.split("\t")
            if lemma not in D:
                D[lemma] = {}
            D[lemma][tag] = word
    return D

def dict2lists(d):
    """
    Convert the given defaultdict object to a list of (lemma,form,feat) tuples.
    d is a list of tuples, where wach tuple has the form (lemma, list) where list has (at most?) 3 dicts with items of the form (feat,form).
    """
    samples_list = []
    for t in d: # t is a tuple
        lemma, forms = t # type(lemma)==str, forms is the list
        assert len(forms) in {1,2,3}
        # train_dict, dev_dict, test_dict = forms
        for dict_set in forms: # train_dict, dev_dict, test_dict = forms
            for k,v in dict_set.items():
                samples_list.append((lemma, v, k))
    return samples_list

def generate_new_datasets(train, dev, test):
    """
    Takes 3 paths for the files, generates
    """
    train_d, dev_d, test_d = read(train), read(dev), read(test)
    ds = [train_d, dev_d, test_d]
    total_d = defaultdict(list)
    # Unite all the forms of the same lemma under a defaultdict value entry
    for d in ds:
        for k,v in d.items():
            total_d[k].append(v)
    lemmas = list(total_d.items())
    n = len(lemmas)
    random.seed(1)
    random.shuffle(lemmas)
    # Now that we shuffled the data, we're ready to split it to 3 new sets, this time with absolute separation between the lemmas.
    train, test, dev = lemmas[:int(0.7 * n)], lemmas[int(0.7 * n):int(0.9 * n)], lemmas[int(0.9 * n):]
    train = dict2lists(train)
    dev = dict2lists(dev)
    test = dict2lists(test) # convert the data to list format (a list of (lemma,form,feat) tuples)
    return train, dev, test, test # the latter is returned twice, once for the gold data file and once for the incomplete file.


def write_dataset(p, dataset, test_mode=False):
    """
    Takes the new dataset, and writes it to the given (new) path.
    """
    with open(p, mode='w', encoding='utf8') as f:
        for sample in dataset:
            lemma, form, feat = sample
            if test_mode:
                f.write(f"{lemma}\t{feat}\n")
            else:
                f.write(f"{lemma}\t{form}\t{feat}\n")

def get_langs_and_paths(data_dir=''):
    """
    Return a list of the languages, and a dictionary of tuples: {lang: (train_path,dev_path,test_paht)}.
    :param data_dir:
    :return:
    """
    train_dirs = ['DEVELOPMENT-LANGUAGES', 'SURPRISE-LANGUAGES']
    test_dir = 'GOLD-TEST'
    train_dirs, test_dir = [os.path.join(data_dir,d) for d in train_dirs], os.path.join(data_dir,test_dir)
    print(f"Requirements: this script must have the same path as the folders of SIGMORPHON (SURPRISE-LANG., GOLD-TEST, etc.). The data folder's name should be {data_dir}.\n")

    test_paths = os.listdir(test_dir)
    langs = [os.path.splitext(p)[0] for p in test_paths]
    test_paths = {l:p for l,p in zip(langs, [os.path.join(test_dir, s) for s in test_paths])}

    dev_families = [f.path for f in os.scandir(train_dirs[0]) if f.is_dir()]
    surprise_families = [f.path for f in os.scandir(train_dirs[1]) if f.is_dir()]

    # dev_families = [e.lower() for e in dev_families]
    # surprise_families = [e.lower() for e in surprise_families]
    dev_families.sort()
    surprise_families.sort()

    develop_paths = {}
    surprise_paths = {}
    test_no_gold_paths = {}
    lang2family = {} # a dictionary that indicates the family of every language
    for family in dev_families+surprise_families:
        for file in os.listdir(family):
            lang, ext = os.path.splitext(file)
            file = os.path.join(family,file)
            if ext=='.trn':
                develop_paths[lang] = file
            elif ext=='.dev':
                surprise_paths[lang] = file
            elif ext=='.tst':
                test_no_gold_paths[lang] = file
            family_name = os.path.split(family)[1]
            if lang not in lang2family: lang2family[lang] = family_name

    files_paths = {k:(develop_paths[k],surprise_paths[k],test_paths[k]) for k in langs}
    return langs, files_paths, lang2family

if __name__ == '__main__':
    print("Requirements: this script must be with the folders of SIGMORPHON (SURPRISE-LANG., GOLD-TEST, etc.).\n This script generates new datasets in the same structure of the old folders.")
    # # region generate_lemma_split
    # train_dirs = ['DEVELOPMENT-LANGUAGES', 'SURPRISE-LANGUAGES']
    # test_dir = 'GOLD-TEST'
    #
    # test_paths = os.listdir(test_dir)
    # langs = [os.path.splitext(p)[0] for p in test_paths]
    # test_paths = {l:p for l,p in zip(langs, [os.path.join(test_dir, s) for s in test_paths])}
    #
    # dev_families = [f.path for f in os.scandir(train_dirs[0]) if f.is_dir()]
    # surprise_families = [f.path for f in os.scandir(train_dirs[1]) if f.is_dir()]
    #
    # dev_families = [e.lower() for e in dev_families]
    # surprise_families = [e.lower() for e in surprise_families]
    # dev_families.sort()
    # surprise_families.sort()
    #
    # develop_paths = {}
    # surprise_paths = {}
    # test_no_gold_paths = {}
    #
    # for family in dev_families+surprise_families:
    #     for file in os.listdir(family):
    #         lang, ext = os.path.splitext(file)
    #         file = os.path.join(family,file)
    #         if ext=='.trn':
    #             develop_paths[lang] = file
    #         elif ext=='.dev':
    #             surprise_paths[lang] = file
    #         elif ext=='.tst':
    #             test_no_gold_paths[lang] = file
    # # print(len(develop_paths))
    # # print(len(surprise_paths))
    # # print(len(test_no_gold_paths))
    #
    #
    # print("Intersections between train, dev & test sets for each of the 90 languages:")
    # for i,lang in enumerate(langs):
    #     result = check_lemma_split(develop_paths[lang], surprise_paths[lang], test_paths[lang]) # returns a triplet
    #     inter1, inter2, inter3 = ["Non-Empty" if e else "Empty" for e in result]
    #     print(f"{i+1}. {lang} => {(inter1, inter2, inter3)}")
    # print()
    #
    #
    # print("Duplicating the original structure of the directories & sub-directories...\n")
    # inputpath = os.getcwd()
    # new_folder = "LEMMA-SPLIT"
    # outputpath = os.path.join(inputpath, new_folder)
    #
    # # Create a new folder "LEMMA-SPLIT", and all the sub-folders like the original structure:
    # for dirpath, dirnames, filenames in os.walk(inputpath):
    #     structure = os.path.join(outputpath, os.path.relpath(dirpath, inputpath))
    #     if not os.path.isdir(structure) and 'idea' not in structure and 'git' not in structure and os.path.join(new_folder, new_folder) not in structure:
    #         os.mkdir(structure)
    #         print(f"Folder {structure}\tcreated.")
    #     else:
    #         print(f"Folder {structure}\texists.")
    # print()
    #
    #
    # print("Generating new datasets:")
    # # For every language, add the "LEMMA-SPLIT" prefix, and write the new datasets (returned from generate_new_datasets) to the files in the corresponding folders
    # # langs = ['aka']
    # for i,lang in enumerate(langs):
    #     paths = [develop_paths[lang], surprise_paths[lang], test_paths[lang], test_no_gold_paths[lang]]
    #     datasets = generate_new_datasets(*paths[:-1])
    #     new_paths = [os.path.join(new_folder, p) for p in paths]
    #
    #     for j,(path,dataset) in enumerate(zip(new_paths,datasets)):
    #         write_dataset(path, dataset, test_mode=(j==3))
    #     print(f"{i+1}. Generated new datasets for {lang}")
    #
    # # endregion generate_lemma_split

    splittype = 'lemma'
    assert splittype in {'form', 'lemma'}
    langs, paths, lang2family = get_langs_and_paths(data_dir=os.path.join('.data',f'{splittype}-split'))
    sizes_df = pandas.DataFrame(columns=['Lang', 'Family', '#Lemmas', '#Forms', '#TrnLemmas', '#DevLemmas', '#TstLemmas', '#TrnForms', '#DevForms', '#TstForms', '#Avg. Par.'])
    for j,lang in enumerate(langs):
        trn, dev, tst = paths[lang]
        if splittype=='form': # no lemma-separation, hence ~same #lemmas is expected in every file.
            lemmas1, lemmas2, lemmas3 = read_all_lemmas(trn), read_all_lemmas(dev), read_all_lemmas(tst)
            totLem = len(set(lemmas1+lemmas2+lemmas3))
            trnLem, devLem, tstLem = '', '', ''
        else:
            lemmas1, lemmas2, lemmas3 = set(read_all_lemmas(trn)), set(read_all_lemmas(dev)), set(read_all_lemmas(tst))
            trnLem, devLem, tstLem = len(lemmas1), len(lemmas2), len(lemmas3)
            totLem = trnLem + devLem + tstLem
        forms1, forms2, forms3 = read(trn), read(dev), read(tst)
        trnForm, devForm, tstForm = sum([len(v) for v in forms1.values()]), sum([len(v) for v in forms2.values()]), sum([len(v) for v in forms3.values()])
        totForm = trnForm + devForm + tstForm
        avg = np.round(totForm / totLem, 2)
        sizes_df.loc[j] = [lang, lang2family[lang], totLem, totForm, trnLem, devLem, tstLem, trnForm, devForm, tstForm, avg]

    sizes_df.to_excel(f"Datasets Sizes.{len(langs)}Langs.{splittype}.xlsx")