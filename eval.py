import argparse
import re
import json
import pprint
import collections
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor_score
from tabulate import tabulate
from utils import *
from rouge import Rouge
import warnings
warnings.simplefilter('ignore')


conch_score = False
if conch_score:
    import sys
    sys.path.append('path/to/conch_codebase')
    from models.conch.open_clip_custom import tokenize, get_tokenizer
    from models.model_conch import conch_coca
    tokenizer = get_tokenizer()
    checkpoint_path = "path/to/conc_ckpt"
    model = conch_coca(checkpoint_path=checkpoint_path).cuda()
    model.eval()

def CONCHScore(gt, res):
    token_ids = tokenize(tokenizer, [gt, res]) # Tokenize with custom tokenizer
    token_ids = token_ids.cuda()
    embed = model.encode_text(token_ids)
    return (embed[0] @ embed[1]).item()


def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def parse_option():
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument('--quilt', type=bool, default=False, help='whether to evaluate on quilt outputs')
    parser.add_argument('--gt', type=str, default="test.json", help='path to groundtruth file', )
    parser.add_argument('--pred', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file', )
    parser.add_argument('--pred_file_parent_path', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file', )
    parser.add_argument('--anchor', type=str, default="", help='path to anchor prediction file, unused except for eval of lengthy preds', )
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

def remove_brackets(text):
    text = re.sub(r'（.*?）', '', text)  # 匹配中文括号及其中的内容
    text = re.sub(r'\(.*?\)', '', text)  # 匹配英文括号及其中的内容
    return text

def evaluate(gt, pred, quilt=False, anchor=None):    
    closed_scores2 = collections.defaultdict(list)
    bleu1_scores = collections.defaultdict(list)
    bleu2_scores = collections.defaultdict(list)
    bleu3_scores = collections.defaultdict(list)
    bleu_scores = collections.defaultdict(list)
    exact_scores = collections.defaultdict(list)
    f1_scores = collections.defaultdict(list)
    meteor_scores = collections.defaultdict(list)
    rougel_scores = collections.defaultdict(list)
    conch_scores = collections.defaultdict(list)
    rouge = Rouge()

    for gt_item, pred_item, anchor_item in zip(gt, pred, anchor if anchor else pred):
        gt_value = gt_item['answer'].lower()
        pred_value = pred_item['text'].lower()
        anchor_value = anchor_item['text'].lower()
        ch = contains_chinese(gt_value)

        gt_value = normalize_word(gt_value)
        pred_value = normalize_word(pred_value)
        anchor_value = normalize_word(anchor_value)

        pred_value = pred_value[:len(anchor_value)]
		
        #pred_value = remove_brackets(pred_value)

        if gt_item['answer_type'] == 'OPEN' or gt_item['answer_type'] == 'other':
            # for open-ended question
            exact_scores['hit'].append(calculate_exactmatch(pred_value, gt_value))
            exact_scores['q_id'].append(pred_item['question_id'])

            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            f1_scores['f1'].append(f1_score)
            f1_scores['precision'].append(precision)
            f1_scores['recall'].append(recall)
            f1_scores['q_id'].append(pred_item['question_id'])

            if ch:
                b1_score = sentence_bleu(references=[list(gt_value)], hypothesis=list(pred_value), weights=(1,))
                b2_score = sentence_bleu(references=[list(gt_value)], hypothesis=list(pred_value), weights=(0.5, 0.5))
                b3_score = sentence_bleu(references=[list(gt_value)], hypothesis=list(pred_value), weights=(1.0 / 3, 1.0 / 3, 1.0 / 3))
                b_score = sentence_bleu(references=[list(gt_value)], hypothesis=list(pred_value)) # default 4 x 0.25
            else:
                b1_score = sentence_bleu(references=[gt_value.split()], hypothesis=pred_value.split(), weights=(1,))
                b2_score = sentence_bleu(references=[gt_value.split()], hypothesis=pred_value.split(), weights=(0.5, 0.5))
                b3_score = sentence_bleu(references=[gt_value.split()], hypothesis=pred_value.split(), weights=(1.0 / 3, 1.0 / 3, 1.0 / 3))
                b_score = sentence_bleu(references=[gt_value.split()], hypothesis=pred_value.split()) # default 4 x 0.25
            
            bleu1_scores['q_id'].append(pred_item['question_id'])
            bleu1_scores['bleu1_score'].append(b1_score)
            bleu2_scores['q_id'].append(pred_item['question_id'])
            bleu2_scores['bleu2_score'].append(b2_score)
            bleu3_scores['q_id'].append(pred_item['question_id'])
            bleu3_scores['bleu3_score'].append(b3_score)
            bleu_scores['q_id'].append(pred_item['question_id'])
            bleu_scores['bleu_score'].append(b_score)
            
            meteor_scores['q_id'].append(pred_item['question_id'])
            if ch:
                meteor_scores['meteor'].append(meteor_score.meteor_score(references=[list(gt_value)], hypothesis=list(pred_value)))
            else:
                meteor_scores['meteor'].append(meteor_score.meteor_score(references=[gt_value.split()], hypothesis=pred_value.split()))

            rougel_scores['q_id'].append(pred_item['question_id'])
            try:
                if ch:
                    rougel_scores['rougel'].append(rouge.get_scores(' '.join(pred_value), ' '.join(gt_value), avg=True)['rouge-l']['f'])
                else:
                    rougel_scores['rougel'].append(rouge.get_scores(pred_value, gt_value, avg=True)['rouge-l']['f'])
            except:
                rougel_scores['rougel'].append(0)

            if not ch and conch_score:
                conch_scores['CONCHScore'].append(CONCHScore(gt_value, pred_value))

        elif gt_item['answer_type'] == 'CLOSED':
            # for close-ended question (Yes/No)
            closed_scores2['q_id'].append(pred_item['question_id'])

            if quilt:
                gt_value = gt_item['yes_no_answer'].lower()

            assert gt_value in ['yes', 'no'], f"assert gt_value in : {pred_item['question_id'], gt_value}"
            answer = gt_value
            # Only keep the first sentence
            #if pred_value.find('.') != -1:
            #    pred_value = pred_value.split('.')[0]

            pred_value = pred_value.replace(',', '')
            words = pred_value.split(' ')
            if 'No' in words or 'not' in words or 'no' in words:
                pred_answer = 'no'
            else:
                pred_answer = 'yes'
            
            if pred_answer == answer:
                closed_scores2['hit'].append(1)
            else:
                closed_scores2['hit'].append(0)
    
    exact_score = sum(exact_scores['hit']) / len(exact_scores['hit'])
    f1_score = sum(f1_scores['f1']) / len(f1_scores['f1'])
    precision = sum(f1_scores['precision']) / len(f1_scores['precision'])
    recall = sum(f1_scores['recall']) / len(f1_scores['recall'])
    closed_score2 = sum(closed_scores2['hit']) / len(closed_scores2['hit']) if len(closed_scores2['hit']) != 0 else 0.0
    bleu1 = sum(bleu1_scores['bleu1_score']) / len(bleu1_scores['bleu1_score'])
    bleu2 = sum(bleu2_scores['bleu2_score']) / len(bleu2_scores['bleu2_score'])
    bleu3 = sum(bleu3_scores['bleu3_score']) / len(bleu3_scores['bleu3_score'])
    bleu = sum(bleu_scores['bleu_score']) / len(bleu_scores['bleu_score'])
    meteor = sum(meteor_scores['meteor']) / len(meteor_scores['meteor'])
    rougel = sum(rougel_scores['rougel']) / len(rougel_scores['rougel'])
    if not ch and conch_score:
        conchs = sum(conch_scores['CONCHScore']) / len(conch_scores['CONCHScore'])
    else:
        conchs = -1

    tab = tabulate(
        [
            ['exact match score', exact_score*100], 
            ['f1 score', f1_score*100], 
            ['precision', precision*100], 
            ['recall', recall*100], 
            ['yes/no accuracy', closed_score2*100],
            ['bleu1', bleu1*100],
            ['bleu2', bleu2*100],
            ['bleu3', bleu3*100],
            ['bleu', bleu*100],
            ['meteor', meteor*100],
            ['rougel', rougel*100],
            ['CONCHScore', conchs*100],
        ], 
        headers=['Metric', 'Performance']
    )

    return tab, bleu

if __name__ == '__main__':
    args = parse_option()

    gt = json.load(open(args.gt, 'r'))
    pred = load_jsonl(args.pred)
    if args.anchor:
        anchor = load_jsonl(args.anchor)
        anchor_ids = [item['question_id'] for item in anchor]

    gt_ids = [item['id'] for item in gt]
    pred_ids = [item['question_id'] for item in pred]
    
    #assert gt_ids == pred_ids, "please make sure pred and gt are exactly matched"

    # perform evaluation
    results = evaluate(gt, pred, quilt=args.quilt, anchor=anchor if args.anchor else None)
    pprint.pprint(results[0])
