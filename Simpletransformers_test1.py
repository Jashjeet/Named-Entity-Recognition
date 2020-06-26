# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:57:28 2020

@author: jashj
"""
        
        
# Conversion to CoNLL      
import json
from spacy.gold import biluo_tags_from_offsets
from spacy.lang.en import English   # or whichever language tokenizer you need

import pandas as pd

annotated_file="C:/Users/jashj/Desktop/Nowigence/Edited jsonl files/Test Train split all files/all_files_train.jsonl"
max_count=1000000

train_data = []


extended_tags = []
extended_entities = []
extended_token = []
sent_num_list=[]
sent_num=0
count = 0
nlp = English()
with open(annotated_file, 'r') as f_in:
    for line in f_in:
        if count >= max_count:
            break

        line = line.strip()
        if len(line) == 0:
            continue

        count += 1
        eg = json.loads(line)
        #print(eg)
        if eg['answer'] == 'accept':
            if eg.get('spans') is None:
                print("ERROR: accept is true but spans are missing, line#{}, line={}".format(count, line))
                continue
            
            doc = nlp(eg['text'])
            for token in doc:
                extended_token.append(token)
            entities = [(span['start'], span['end'], span['label']) for span in eg['spans']]
            tags = biluo_tags_from_offsets(doc, entities)
            extended_tags.extend(tags)             
            extended_entities.extend(entities)
            train_res = (eg['text'], {'entities': entities})
            
            sent_num_list.extend([sent_num]*len(tags))
            sent_num+=1
            #train_res = example_to_train_res(eg)
            train_data.append(train_res)
            #print(train_res)
            #print(train_res[1])

x=set(extended_tags)
label_list=list(x)


zipped=list(zip(sent_num_list,extended_token,extended_tags))

# Converting to Dataframe
train_df = pd.DataFrame(zipped, columns=['sentence_id', 'words', 'labels'])


# Taking a sample of 12 sentences. To run on the complete file, comment the below line
train_df=train_df[:343]



# These are the changes in default hyperparameters that are passed in NERModel
args = {
    "output_dir": "outputs/",
    "cache_dir": "cache_dir/",

    "fp16": True,
    "fp16_opt_level": "O1",
    "max_seq_length": 128,
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "eval_batch_size": 8,
    "num_train_epochs": 2,
    "weight_decay": 0,
    "learning_rate": 4e-5,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,

    "logging_steps": 10,
    "save_steps": 200,

    "overwrite_output_dir": False,
    "reprocess_input_data": False,
    "evaluate_during_training": False,

#    "process_count": cpu_count() - 2 if cpu_count() > 2 else 1,
    "process_count": 2,
    "n_gpu": 1,
}














from simpletransformers.ner import NERModel
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# Create a NERModel
model = NERModel('bert', 'bert-base-cased', use_cuda=True, labels=label_list, args=args)

model.train_model(train_df)


sentences = ["Mary M. Mrdutt, M.D., from the Baylor Scott & White Memorial Hospital in Temple, Texas, and colleagues prospectively measured frailty in elective surgery patients in a health care system.", 
             "To be clear, there are currently no legal requirements for any cosmetic manufacturer marketing products to American consumers to test their products for safety"]
predictions, raw_outputs = model.predict(sentences)

print(predictions)