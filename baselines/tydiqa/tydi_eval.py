# coding=utf-8
# Copyright 2020 The Google Research Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
r"""Official evaluation script for the TyDi QA primary tasks.

The primary tasks are the Passage Selection Task (SelectP) and the Minimal
Answer Span Task Task (AnsSpan). This script is *not* used for the secondary
task, the SQuAD-compatible gold Passage (GoldP) task.

  ------------------------------------------------------------------------------

  Example usage:

  tydi_eval --gold_path=<path-to-gold-files> --predictions_path=<path_to_jsonl>

  This will compute both the official byte-level F1 scores, recall@precision
  tables for both passage and minimal answers (if the optional answer scores are
  provided), and also breakdown per language.

  Note that R@P are only meaningful if your model populates the score fields
  of the prediction JSON format (which is not required).

  gold_path should point to a single N way annotated dev data in the
  original download format (gzipped jsonlines) or jsonlines.

  predictions_path should point to a jsonl file (one json object per line),
  where each line contains the predictions in the format given below.

  ------------------------------------------------------------------------------

  Prediction format (written on multiple lines here for clarity, but each
  prediction should be a single line in your output file):

  {
    'example_id': -2226525965842375672,
    'passage_answer_index': 2,
    'passage_answer_score': 13.5,
    'minimal_answer': {'start_byte_offset': 64206, 'end_byte_offset': 64280},
    'minimal_answer_score': 26.4,
    'yes_no_answer': 'NONE'
  }

  The prediction format mirrors the annotation format in defining each passage
  or minimal answer span both in terms of byte offsets.

  If start_byte_offset >= 0 and end_byte_offset >=0, use byte offsets,
    else no span is defined (null answer).

  The minimal answer metric takes both minimal answer spans, and the yes/no
  answer into account. If the 'minimal_answers' list contains any non/null
  spans, then 'yes_no_answer' should be set to 'NONE'.

  -----------------------------------------------------------------------------

  Metrics:

  Each prediction should be provided with a passage answer score, and a minimal
  answers score. At evaluation time, the evaluation script will find a score
  threshold at which F1 is maximized. All predictions with scores below this
  threshold are ignored (assumed to be null). If the score is not provided,
  the evaluation script considers all predictions to be valid. The script
  will also output the maximum recall at precision points of >= 0.5, >= 0.75,
  and >= 0.9.

  Key methods:
    Scoring passage answer candidates: score_passage_answer()
    Scoring minimal answer candidates: score_minimal_answer(),
                                       eval_utils.compute_partial_match_scores()
    Computing language-wise F1: compute_macro_f1()
    Averaging over non-English languages: main()

"""

import collections
import json
import os
import pickle

from absl import app
from absl import flags
from absl import logging
import eval_utils

flags.DEFINE_string(
    'gold_path', None, 'Path to the gzip JSONL data. For '
    'multiple files, should be a glob '
    'pattern (e.g. "/path/to/files-*"')

flags.DEFINE_string('predictions_path', None,
                    'Path to JSONL file of predictions.')


flags.DEFINE_bool('verbose', True,
                  'Print some sample predictions with its socres.')

flags.DEFINE_bool(
    'cache_gold_data', False,
    'Whether to cache gold data in Pickle format to speed up '
    'multiple evaluations.')

flags.DEFINE_bool('pretty_print', False, 'Whether to pretty print output.')

flags.DEFINE_integer(
    'passage_non_null_threshold', 2,
    'Require this many non-null passage answer annotations '
    'to count gold as containing a passage answer.')

flags.DEFINE_integer(
    'minimal_non_null_threshold', 2,
    'Require this many non-null minimal answer annotations '
    'to count gold as containing a minimal answer.')

FLAGS = flags.FLAGS


def score_passage_answer(gold_label_list, pred_label,
                         passage_non_null_threshold):
  """Scores a passage answer as correct or not.

  1) First decide if there is a gold passage answer with
     FLAGS.passage_non_null_threshold.
  2) The prediction will get a match if:
     a. There is a gold passage answer.
     b. The prediction span match exactly with *one* of the non-null gold
        passage answer index.

  Args:
    gold_label_list: A list of TyDiLabel, could be None.
    pred_label: A single TyDiLabel, could be None.
    passage_non_null_threshold: See FLAGS.passage_non_null_threshold.

  Returns:
    gold_has_answer, pred_has_answer, is_correct, score
  """
  gold_has_answer = eval_utils.gold_has_passage_answer(
      gold_label_list, passage_non_null_threshold)

  if pred_label is None:
    return gold_has_answer, not gold_has_answer, False, 0

  pred_has_answer = pred_label.passage_answer_index >= 0

  is_correct = False
  score = pred_label.passage_score

  # Both sides are non-null spans.
  if gold_has_answer and pred_has_answer:
    for gold_label in gold_label_list:
      # while the voting results indicate there is an passage answer, each
      # annotator might still say there is no passage answer.
      if gold_label.passage_answer_index < 0:
        continue

      if gold_label.passage_answer_index == pred_label.passage_answer_index:
        is_correct = True
        break

  return gold_has_answer, pred_has_answer, is_correct, score


def score_minimal_answer(gold_label_list, pred_label,
                         minimal_non_null_threshold):
  """Scores a minimal answer.

  Outputs score against gold label that gives max F1.

  First decide if there is a gold minimal answer with
  FLAGS.minimal_non_null_threshold.
  If any of the gold label has "yes", or "no", and pred label predicted it
  correctly, than precision, recall, f1 is all 1.0.

  Args:
    gold_label_list: A list of TyDiLabel.
    pred_label: A single TyDiLabel.
    minimal_non_null_threshold: See FLAGS.minimal_non_null_threshold.

  Returns:
    gold_has_answer, pred_has_answer, (precision, recall, f1), score
  """

  # There is a gold minimal answer if gold_label_list not empty and non null
  # answers is over the threshold (sum over annotators).
  gold_has_answer = eval_utils.gold_has_minimal_answer(
      gold_label_list, minimal_non_null_threshold)

  if pred_label is None:
    return gold_has_answer, not gold_has_answer, (0, 0, 0), 0

  # There is a predicted minimal answer if the predicted minimal label span
  # is non-null or we have a specific predicted label (such as yes/no).
  pred_has_answer = ((not pred_label.minimal_answer_span.is_null_span()) or
                     pred_label.yes_no_answer != 'none')

  # score is optional.
  score = pred_label.minimal_score
  # We find the closest (highest scoring) match between the system's predicted
  # minimal answer and one of the three gold annotations.
  max_f1 = 0.0
  max_precision = 0.0
  max_recall = 0.0

  # Both sides have minimal answers, which contains yes/no questions.
  if gold_has_answer and pred_has_answer:
    if pred_label.yes_no_answer != 'none':  # System predicted a yes/no answer.
      for gold_label in gold_label_list:
        if pred_label.yes_no_answer == gold_label.yes_no_answer:
          max_f1 = 1.0
          max_precision = 1.0
          max_recall = 1.0
          break
    else:
      for gold_label in gold_label_list:
        if gold_label.minimal_answer_span.is_null_span():
          continue
        # Compute the *micro-F1* (a partial match score for this example).
        # We also compute a language-wise *macro-F1* later.
        precision, recall, f1 = eval_utils.compute_partial_match_scores(
            gold_label.minimal_answer_span, pred_label.minimal_answer_span)
        if f1 > max_f1:
          max_f1 = f1
          max_precision = precision
          max_recall = recall

  return (gold_has_answer, pred_has_answer,
          (max_precision, max_recall, max_f1), score)


def byte_slice(text, start, end):
  byte_str = bytes(text, 'utf-8')
  # byte_str = text
  try:
    return str(byte_str[start:end].decode('utf-8'))
  except UnicodeDecodeError:
      return str(byte_str[start:end])


def save_minimal(gold_annotation_dict, pred_dict, lang):
  gold_id_set = set(gold_annotation_dict.keys())
  pred_id_set = set(pred_dict.keys())

  unpredicted = gold_id_set - pred_id_set
  unexpected = pred_id_set - gold_id_set

  passage_answer_stats = []
  minimal_answer_stats = []
  example_count = 0
  log_rec = []
  passage_id = {}
  minimal_id = {}
  min_f1_1={}
  min_f1={}
  p_f1={}
  real_examples = 0

  for example_id in gold_id_set:
    example_count += 1
    gold = gold_annotation_dict[example_id]
    pred = pred_dict.get(example_id)
    passage_answer_stats.append(
        score_passage_answer(gold, pred, FLAGS.passage_non_null_threshold))
    minimal_answer_stats.append(
        score_minimal_answer(gold, pred, FLAGS.minimal_non_null_threshold))
    passage_id[example_id]=score_passage_answer(gold, pred, FLAGS.passage_non_null_threshold)
    minimal_id[example_id]=score_minimal_answer(gold, pred, FLAGS.minimal_non_null_threshold)

    if not FLAGS.verbose:
      continue
    if pred is None:
      continue
    pred_min_start = pred.minimal_answer_span.start_byte_offset
    pred_min_end = pred.minimal_answer_span.end_byte_offset
    gold_min_start = gold[0].minimal_answer_span.start_byte_offset
    gold_min_end = gold[0].minimal_answer_span.end_byte_offset
    if gold_min_start >= 0:
      real_examples+=1
      logging.info('---')
      logging.info(gold[0].example_id)
      logging.info(gold[0].question_text)
      logging.info('gold offsets %d, %d', gold_min_start, gold_min_end)
      logging.info('pred offsets %d, %d', pred_min_start, pred_min_end)
      logging.info('gold answer: (%s)',
                   byte_slice(gold[0].plaintext, gold_min_start, gold_min_end))
      logging.info('pred answer: (%s)',
                   byte_slice(gold[0].plaintext, pred_min_start, pred_min_end))
      logging.info('score %.2f', minimal_answer_stats[-1][-1])
      logging.info('f1: %.2f, p: %.2f, r: %.2f',
                   minimal_answer_stats[-1][-2][2],
                   minimal_answer_stats[-1][-2][0],
                   minimal_answer_stats[-1][-2][1])
      #for custom log dump
      p_f1[gold[0].example_id]=passage_answer_stats[-1][-1]
      min_f1_1[gold[0].example_id]=minimal_answer_stats[-1][-2][2]
      min_f1[gold[0].example_id]={'q':gold[0].question_text, 
                                  'f1':minimal_answer_stats[-1][-2][2]}
      p_f1[gold[0].example_id]={'q':gold[0].question_text, 
                                  'f1':minimal_answer_stats[-1][-2][2]}
      log_r = "'{}'\nq_text: {}\ngold_offset: {}, {}\npred_offset: {} {}\n\
gold_answer: {}\npred_answer: {}\nscore: {}\nf1: {:.2f} p: {:.2f} r: {:.2f}\n".format(
                gold[0].example_id, gold[0].question_text, 
                gold_min_start, gold_min_end,
                pred_min_start, pred_min_end,
                byte_slice(gold[0].plaintext, gold_min_start, gold_min_end),
                byte_slice(gold[0].plaintext, pred_min_start, pred_min_end),
                minimal_answer_stats[-1][-1],
                minimal_answer_stats[-1][-2][2],
                minimal_answer_stats[-1][-2][0],
                minimal_answer_stats[-1][-2][1]
                )
      log_rec.append(log_r)
  # dump log into pickle
  with open(FLAGS.log_dir+'-'+lang+'.pickle', 'wb') as f:
    pickle.dump(log_rec, f)
    logging.info("log data saved in {}".format(FLAGS.log_dir+'-'+lang+'.pickle'))
  
  # with open(FLAGS.log_dir+'-'+lang+'-f1.pickle', 'wb') as f:
  #   pickle.dump(min_f1, f)
  #   logging.info("f1 data saved in {}".format(FLAGS.log_dir+'-'+lang+'-f1.pickle'))
  # logging.info('example_count:{}, real_examples:{}'.format(example_count, real_examples))

  return p_f1, min_f1_1

  # use the 'score' column, which is last
  # passage_answer_stats.sort(key=lambda x: x[-1], reverse=True)
  # minimal_answer_stats.sort(key=lambda x: x[-1], reverse=True)
  

def score_answers(gold_annotation_dict, pred_dict):
  """Scores all answers for all documents.

  Args:
    gold_annotation_dict: a dict from example id to list of `TyDiLabel`s.
    pred_dict: a dict from example id to list of `TyDiLabel`s.

  Returns:
    passage_answer_stats: List of scores for passage answers.
    minimal_answer_stats: List of scores for minimal answers.
  """
  gold_id_set = set(gold_annotation_dict.keys())
  pred_id_set = set(pred_dict.keys())

  unpredicted = gold_id_set - pred_id_set
  unexpected = pred_id_set - gold_id_set
  if unpredicted:
    logging.warning('Predictions missing for %d examples.', len(unpredicted))
    logging.info('  Missing ids: %s', sorted(unpredicted))
  if unexpected:
    logging.warning(
        'Found predictions for %d examples that do not appear in the gold data.',
        len(unexpected))
    logging.info('  Unexpected ids: %s', sorted(unexpected))

  passage_answer_stats = []
  minimal_answer_stats = []
  example_count = 0
  log_rec = []
  passage_id = []
  minimal_id = []
  min_f1={}
  for example_id in gold_id_set:
    example_count += 1
    gold = gold_annotation_dict[example_id]
    pred = pred_dict.get(example_id)
    passage_answer_stats.append(
        score_passage_answer(gold, pred, FLAGS.passage_non_null_threshold))
    passage_id.append((example_id,score_passage_answer(gold, pred, FLAGS.passage_non_null_threshold)))
    minimal_answer_stats.append(
        score_minimal_answer(gold, pred, FLAGS.minimal_non_null_threshold))
    minimal_id.append((example_id,score_minimal_answer(gold, pred, FLAGS.minimal_non_null_threshold)))

    if not FLAGS.verbose:
      continue
    if pred is None:
      continue
    pred_min_start = pred.minimal_answer_span.start_byte_offset
    pred_min_end = pred.minimal_answer_span.end_byte_offset
    gold_min_start = gold[0].minimal_answer_span.start_byte_offset
    gold_min_end = gold[0].minimal_answer_span.end_byte_offset
    if gold_min_start >= 0:
      logging.info('---')
      logging.info(gold[0].example_id)
      logging.info(gold[0].question_text)
      logging.info('gold offsets %d, %d', gold_min_start, gold_min_end)
      logging.info('pred offsets %d, %d', pred_min_start, pred_min_end)
      logging.info('gold answer: (%s)',
                   byte_slice(gold[0].plaintext, gold_min_start, gold_min_end))
      logging.info('pred answer: (%s)',
                   byte_slice(gold[0].plaintext, pred_min_start, pred_min_end))
      logging.info('score %.2f', minimal_answer_stats[-1][-1])
      logging.info('f1: %.2f, p: %.2f, r: %.2f',
                   minimal_answer_stats[-1][-2][2],
                   minimal_answer_stats[-1][-2][0],
                   minimal_answer_stats[-1][-2][1])
      #for custom log dump
      min_f1[gold[0].example_id]={'q':gold[0].question_text, 
                                  'f1':minimal_answer_stats[-1][-2][2]}
      log_r = "'{}'\nq_text: {}\ngold_offset: {}, {}\npred_offset: {} {}\n\
gold_answer: {}\npred_answer: {}\nscore: {}\nf1: {:.2f} p: {:.2f} r: {:.2f}\n".format(
                gold[0].example_id, gold[0].question_text, 
                gold_min_start, gold_min_end,
                pred_min_start, pred_min_end,
                byte_slice(gold[0].plaintext, gold_min_start, gold_min_end),
                byte_slice(gold[0].plaintext, pred_min_start, pred_min_end),
                minimal_answer_stats[-1][-1],
                minimal_answer_stats[-1][-2][2],
                minimal_answer_stats[-1][-2][0],
                minimal_answer_stats[-1][-2][1]
                )
      log_rec.append(log_r)
  # dump log into pickle
  # with open(FLAGS.log_dir, 'wb') as f:
  #   pickle.dump(log_rec, f)
  #   logging.info("log data saved in {}".format(FLAGS.log_dir))
  
  # with open(FLAGS.log_dir+'-f1.pickle', 'wb') as f:
  #   pickle.dump(min_f1, f)
  #   logging.info("f1 data saved in {}".format(FLAGS.log_dir+'-f1.pickle'))
  # use the 'score' column, which is last
  passage_answer_stats.sort(key=lambda x: x[-1], reverse=True)
  passage_id.sort(key=lambda x: x[1][-1], reverse=True)
  minimal_answer_stats.sort(key=lambda x: x[-1], reverse=True)
  minimal_id.sort(key=lambda x: x[1][-1], reverse=True)
  return passage_answer_stats, minimal_answer_stats, passage_id, minimal_id


def compute_macro_f1(answer_stats, prefix=''):
  """Computes F1, precision, recall for a list of answer scores.

  This computes the *language-wise macro F1*. For minimal answers,
  we also compute a partial match score that uses F1, which would be
  included in this computation via `answer_stats`.

  Args:
    answer_stats: List of per-example scores.
    prefix (''): Prefix to prepend to score dictionary.

  Returns:
    Dictionary mapping measurement names to scores.
  """

  has_gold, has_pred, f1, _ = list(zip(*answer_stats))

  macro_precision = eval_utils.safe_divide(sum(f1), sum(has_pred))
  macro_recall = eval_utils.safe_divide(sum(f1), sum(has_gold))
  macro_f1 = eval_utils.safe_divide(
      2 * macro_precision * macro_recall,
      macro_precision + macro_recall)

  return collections.OrderedDict({
      prefix + 'n': len(answer_stats),
      prefix + 'f1': macro_f1,
      prefix + 'precision': macro_precision,
      prefix + 'recall': macro_recall
  })


def compute_final_f1(passage_answer_stats, minimal_answer_stats):
  """Computes overall F1 given passage and minimal answers, ignoring scores.

  Note: this assumes that the answers have been thresholded based on scores.

  Arguments:
     passage_answer_stats: List of passage answer scores.
     minimal_answer_stats: List of minimal answer scores.


  Returns:
     Dictionary of name (string) -> score.
  """
  scores = compute_macro_f1(passage_answer_stats, prefix='passage-answer-')
  scores.update(compute_macro_f1(
      minimal_answer_stats, prefix='minimal-answer-'))
  return scores


def compute_pr_curves(answer_stats, targets=None):
  """Computes PR curve and returns R@P for specific targets.

  The values are computed as follows: find the (precision, recall) point
  with maximum recall and where precision > target.

  This is only relevant if you return the system scores in your predictions.
  You may find this useful when attempting to tune the threshold for your
  system on the dev set before requesting an evaluation on the test set
  via the leaderboard.

  Arguments:
    answer_stats: List of statistic tuples from the answer scores.
    targets (None): List of precision thresholds to target.

  Returns:
    List of table with rows: [target, r, p, score].
  """
  total_f1 = 0
  total_has_pred = 0
  total_has_gold = 0

  # Count the number of gold annotations.
  for has_gold, _, _, _ in answer_stats:
    total_has_gold += has_gold

  # Keep track of the point of maximum recall for each target.

  max_recall = [0 for _ in targets]
  max_precision = [0 for _ in targets]
  max_scores = [None for _ in targets]

  # Only keep track of unique thresholds in this dictionary.
  scores_to_stats = collections.OrderedDict()

  # Loop through every possible threshold and compute precision + recall.

  for has_gold, has_pred, is_correct_or_f1, score in answer_stats:
    if isinstance(is_correct_or_f1, tuple):
      _, _, f1 = is_correct_or_f1
    else:
      f1 = is_correct_or_f1
    total_f1 += f1
    total_has_pred += has_pred

    precision = eval_utils.safe_divide(total_f1, total_has_pred)
    recall = eval_utils.safe_divide(total_f1, total_has_gold)

    # If there are any ties, this will be updated multiple times until the
    # ties are all counted.
    scores_to_stats[score] = [precision, recall]



  best_f1 = 0.0
  best_precision = 0.0
  best_recall = 0.0
  best_threshold = 0.0

  for threshold, (precision, recall) in scores_to_stats.items():
    # Match the thresholds to the find the closest precision above some target.
    for t, target in enumerate(targets):
      if precision >= target and recall > max_recall[t]:
        max_recall[t] = recall
        max_precision[t] = precision
        max_scores[t] = threshold

    # Compute optimal threshold.
    f1 = eval_utils.safe_divide(2 * precision * recall, precision + recall)
    if f1 > best_f1:
      best_f1 = f1
      best_precision = precision
      best_recall = recall
      best_threshold = threshold

  return ((best_f1, best_precision, best_recall, best_threshold),
          list(zip(targets, max_recall, max_precision, max_scores)))


def compute_id_f1(ex_ids, answer_stats, targets=None):
  """Computes PR curve and returns R@P for specific targets.

  The values are computed as follows: find the (precision, recall) point
  with maximum recall and where precision > target.

  This is only relevant if you return the system scores in your predictions.
  You may find this useful when attempting to tune the threshold for your
  system on the dev set before requesting an evaluation on the test set
  via the leaderboard.

  Arguments:
    answer_stats: List of statistic tuples from the answer scores.
    targets (None): List of precision thresholds to target.

  Returns:
    List of table with rows: [target, r, p, score].
  """
  total_f1 = 0
  total_has_pred = 0
  total_has_gold = 0

  # Count the number of gold annotations.
  for has_gold, _, _, _ in answer_stats:
    total_has_gold += has_gold

  # Keep track of the point of maximum recall for each target.

  max_recall = [0 for _ in targets]
  max_precision = [0 for _ in targets]
  max_scores = [None for _ in targets]

  # Only keep track of unique thresholds in this dictionary.
  scores_to_stats = collections.OrderedDict()

  # Loop through every possible threshold and compute precision + recall.

  for has_gold, has_pred, is_correct_or_f1, score in answer_stats:
    if isinstance(is_correct_or_f1, tuple):
      _, _, f1 = is_correct_or_f1
    else:
      f1 = is_correct_or_f1
    total_f1 += f1
    total_has_pred += has_pred

    precision = eval_utils.safe_divide(total_f1, total_has_pred)
    recall = eval_utils.safe_divide(total_f1, total_has_gold)

    # If there are any ties, this will be updated multiple times until the
    # ties are all counted.
    scores_to_stats[score] = [precision, recall]



  best_f1 = 0.0
  best_precision = 0.0
  best_recall = 0.0
  best_threshold = 0.0
  all_ids={}
  count=0
  for threshold, (precision, recall) in scores_to_stats.items():
    # Match the thresholds to the find the closest precision above some target.
    for t, target in enumerate(targets):
      if precision >= target and recall > max_recall[t]:
        max_recall[t] = recall
        max_precision[t] = precision
        max_scores[t] = threshold

    # Compute optimal threshold.
    f1 = eval_utils.safe_divide(2 * precision * recall, precision + recall)
    all_ids[ex_ids[count]]=f1
    count+=1
    if f1 > best_f1:
      best_f1 = f1
      best_precision = precision
      best_recall = recall
      best_threshold = threshold

  return all_ids



def print_r_at_p_table(answer_stats):
  """Pretty prints the R@P table for default targets."""
  opt_result, pr_table = compute_pr_curves(
      answer_stats, targets=[0.5, 0.75, 0.9])
  f1, precision, recall, threshold = opt_result
  print('Optimal threshold: {:.5}'.format(threshold))
  print(' F1     /  P      /  R')
  print('{: >7.2%} / {: >7.2%} / {: >7.2%}'.format(f1, precision, recall))
  for target, recall, precision, row in pr_table:
    print('R@P={}: {:.2%} (actual p={:.2%}, score threshold={:.4})'.format(
        target, recall, precision, row))


def get_metrics_as_dict(gold_path, prediction_path):
  """Library version of the end-to-end evaluation.

  Arguments:
    gold_path: Path to a single JSONL data. Could be gzipped or not.
    prediction_path: Path to the JSONL file of prediction data.

  Returns:
    metrics: A dictionary mapping string names to metric scores.
  """

  tydi_gold_dict = eval_utils.read_annotation(gold_path)
  tydi_pred_dict = eval_utils.read_prediction_jsonl(prediction_path)

  passage_answer_stats, minimal_answer_stats = score_answers(
      tydi_gold_dict, tydi_pred_dict)

  return get_metrics_with_answer_stats(
      passage_answer_stats, minimal_answer_stats)


def get_metrics_with_answer_stats(passage_answer_stats, minimal_answer_stats):
  """Generate metrics dict using passage and minimal answer stats."""

  def _get_metric_dict(answer_stats, prefix=''):
    """Compute all metrics for a set of answer statistics."""
    opt_result, pr_table = compute_pr_curves(
        answer_stats, targets=[0.5, 0.75, 0.9])
    f1, precision, recall, threshold = opt_result
    metrics = collections.OrderedDict({
        'best-threshold-f1': f1,
        'best-threshold-precision': precision,
        'best-threshold-recall': recall,
        'best-threshold': threshold,
    })
    for target, recall, precision, _ in pr_table:
      metrics['recall-at-precision>={:.2}'.format(target)] = recall
      metrics['precision-at-precision>={:.2}'.format(target)] = precision

    # Add prefix before returning.
    return dict([(prefix + k, v) for k, v in metrics.items()])

  metrics = _get_metric_dict(passage_answer_stats, 'passage-')
  metrics.update(_get_metric_dict(minimal_answer_stats, 'minimal-'))
  return metrics


def get_latex_str(f1, precision, recall):
  return '\\fpr' + '{' + ('%.1f' % (f1 * 100)) + '}{' + (
      '%.1f' % (precision * 100)) + '}{' + ('%.1f' % (recall * 100)) + '}'


def main(_):
  # logging.get_absl_handler().use_absl_log_file(FLAGS.predictions_path.split('/')[-1])
  cache_path = os.path.join(os.path.dirname(FLAGS.gold_path), 'cache')
  if FLAGS.cache_gold_data and os.path.exists(cache_path):
    logging.info('Reading from cache: %s', format(cache_path))
    tydi_gold_dict = pickle.load(open(cache_path, 'r'))  # pytype: disable=wrong-arg-types
  else:
    tydi_gold_dict = eval_utils.read_annotation(FLAGS.gold_path)
    if FLAGS.cache_gold_data:
      logging.info('Caching gold data for future to: %s', format(cache_path))
      pickle.dump(tydi_gold_dict, open(cache_path, 'w'))  # pytype: disable=wrong-arg-types
  total_ans_count = 0
  count = 0

  for ans in tydi_gold_dict.values():
    count += 1
    gold_has_answer = eval_utils.gold_has_minimal_answer(
        ans, FLAGS.minimal_non_null_threshold)
    total_ans_count += gold_has_answer

  logging.info('%d examples have minimal answers', total_ans_count)
  logging.info('*' * 40)
  tydi_pred_dict = eval_utils.read_prediction_jsonl(FLAGS.predictions_path)

  per_lang_gold = {}
  per_lang_pred = {}

  for ex_id, ex in tydi_gold_dict.items():
    if ex[0].language in per_lang_gold:
      per_lang_gold[ex[0].language][ex_id] = ex
    else:
      per_lang_gold[ex[0].language] = {ex_id: ex}
  for ex_id, ex in tydi_pred_dict.items():
    if ex.language in per_lang_pred:
      per_lang_pred[ex.language][ex_id] = ex
    else:
      per_lang_pred[ex.language] = {ex_id: ex}

  macro_avg_passage_scores = ([], [], [])
  macro_avg_minimal_scores = ([], [], [])

  language_list = [
      'english', 'arabic', 'bengali', 'finnish', 'indonesian', 'japanese',
      'swahili', 'korean', 'russian', 'telugu', 'thai', 'english-arabic',
      'english-bengali', 'english-swahili', 'english-korean'
  ]
  all_min={}
  all_pass={}
  for lang in language_list:
    if lang in per_lang_pred:
      passage_answer_stats, minimal_answer_stats, passage_id, minimal_id = score_answers(
          per_lang_gold.get(lang, {}), per_lang_pred[lang])
      ps,ms=save_minimal(per_lang_gold.get(lang, {}), per_lang_pred[lang],lang)

      # Passage selection task
      opt_result, _ = compute_pr_curves(passage_answer_stats,  targets=[0.5])
      exids,statss = map(list, zip(*passage_id))
      pass_f1 = compute_id_f1(exids, statss, targets=[0.5])
      maxval=max(pass_f1.values())
      f1, precision, recall, _ = opt_result
      if lang != 'english':
        macro_avg_passage_scores[0].append(f1)
        macro_avg_passage_scores[1].append(precision)
        macro_avg_passage_scores[2].append(recall)
      print('Passage & ' + lang + ' & ' + get_latex_str(f1, precision, recall))

      # Minimal answer span task
      opt_result, _ = compute_pr_curves(minimal_answer_stats, targets=[0.5])
      exids,statss = map(list, zip(*minimal_id))
      min_f1 = compute_id_f1(exids, statss, targets=[0.5])
      maxval=max(min_f1.values())
      f1, precision, recall, _ = opt_result
      if lang != 'english':
        macro_avg_minimal_scores[0].append(f1)
        macro_avg_minimal_scores[1].append(precision)
        macro_avg_minimal_scores[2].append(recall)
      print('Minimal Answer & ' + lang + ' & ' +
            get_latex_str(f1, precision, recall))
      
      
      all_min[lang] = min_f1
      all_pass[lang] = pass_f1

      if FLAGS.pretty_print:
        print('*' * 20)
        print(lang)
        print('Language: %s (%d)' % (lang, len(per_lang_gold.get(lang, {}))))
        print('*' * 20)
        print('PASSAGE ANSWER R@P TABLE:')
        print_r_at_p_table(passage_answer_stats)
        print('*' * 20)
        print('MINIMAL ANSWER R@P TABLE:')
        print_r_at_p_table(minimal_answer_stats)
      else:
        metrics = get_metrics_with_answer_stats(passage_answer_stats,
                                                minimal_answer_stats)
        print(json.dumps(metrics))
  import pickle
  p_file = 'passage|'+'|'.join(FLAGS.predictions_path.split('/')[1:])+'.pickle'
  m_file = 'minimal|'+'|'.join(FLAGS.predictions_path.split('/')[1:])+'.pickle'
  print(p_file, m_file)
  # with open('../f1_files/'+p_file,'wb') as f:
  #     pickle.dump(all_pass,f)
  # with open('../f1_files/'+m_file,'wb') as f:
  #     pickle.dump(all_min,f)
  # print('Total # examples in gold: %d, # ex. in pred: %d (including english)' %
  #       (len(tydi_gold_dict), len(tydi_pred_dict)))

  f1_list, precision_list, recall_list = macro_avg_passage_scores
  print('*** Macro Over %d Languages, excluding English **' % len(f1_list))
  avg_passage_f1 = eval_utils.safe_average(f1_list)
  avg_passage_recall = eval_utils.safe_average(recall_list)
  avg_passage_precision = eval_utils.safe_average(precision_list)
  print('Passage F1:%.3f P:%.3f R:%3f' %
        (avg_passage_f1, avg_passage_precision, avg_passage_recall))
  print(get_latex_str(
      avg_passage_f1, avg_passage_precision, avg_passage_recall))

  f1_list, precision_list, recall_list = macro_avg_minimal_scores

  avg_minimal_f1 = eval_utils.safe_average(f1_list)
  avg_minimal_recall = eval_utils.safe_average(recall_list)
  avg_minimal_precision = eval_utils.safe_average(precision_list)
  print('Minimal F1:%.3f P:%.3f R:%3f' %
        (avg_minimal_f1, avg_minimal_precision, avg_minimal_recall))
  print(get_latex_str(
      avg_minimal_f1, avg_minimal_precision, avg_minimal_recall))
  print('*** / Aggregate Scores ****')

  aggregate_metrics = {'avg_passage_f1': avg_passage_f1,
                       'avg_passage_recall': avg_passage_recall,
                       'avg_passage_precision': avg_passage_precision,
                       'avg_minimal_f1': avg_minimal_f1,
                       'avg_minimal_recall': avg_minimal_recall,
                       'avg_minimal_precision': avg_minimal_precision}
  print(json.dumps(aggregate_metrics))

if __name__ == '__main__':
  flags.mark_flag_as_required('gold_path')
  flags.mark_flag_as_required('predictions_path')
  app.run(main)
