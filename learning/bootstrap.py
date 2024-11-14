#!/usr/bin/env python3

"""Implements the conjecture-prove bootstrapping learning loop."""

import asyncio
import os
import io
import json
import datetime
import random
import yaml

import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from tqdm import tqdm

import peano
import worker
from worker import StudentResult  # noqa
from hindsight import HindsightExample  # noqa
from util import format_blocks_with_indent, sample_batch, setup_mle_logger, value_color, save_json, load_final_goals
from conjecture import AgentLM, Context, sample_conjecture
from proofsearch import make_agent

from mle_logging import MLELogger

log = logging.getLogger(__name__)

FAIL = "fail"


DISTRIBUTED = os.environ.get('DISTRIBUTED', False)


def submit_task(agent_dump: bytes, theory: worker.BackgroundTheory, statement: str):
    if DISTRIBUTED:
        return worker.try_prove.apply_async((agent_dump, theory, statement))
    else:
        return worker.try_prove.run(agent_dump, theory, statement)


def get_task_result(task):
    if DISTRIBUTED:
        return task.get()
    else:
        return task



async def teacher_loop(cfg: DictConfig, mle_log: MLELogger):
    log.info('Running in %s', 'distributed mode.' if DISTRIBUTED else 'single-process mode.')
    agent = make_agent(cfg, mle_log)

    # load goals from file and format them
    # FIXME(f.srambical): check whether the goal set is correctly formatted (check the first few finetuning examples)
    final_goals_formatted, final_solutions = load_final_goals(os.path.join(os.path.dirname(__file__), '../goals', cfg.goals + '.json'))
    final_goals = ["Conj:(hard) " + g for g in final_goals_formatted]

    # load proper validation set 
    val_goals_formatted, val_solutions = [], []
    if cfg.get('val_goals'):
        if os.path.isfile(os.path.join(os.path.dirname(__file__), '../goals', cfg.val_goals + ".json")):
            val_goals_formatted, val_solutions = load_final_goals(os.path.join(os.path.dirname(__file__), '../goals', cfg.val_goals + ".json"))
        else:
            for f in os.listdir(os.path.join(os.path.dirname(__file__), '../goals')):
                if f.startswith(cfg.val_goals) and f.endswith('.json'):
                    val_goals_iter, val_solutions_iter = load_final_goals(os.path.join(os.path.dirname(__file__), '../goals', f))
                    val_goals_formatted.extend(val_goals_iter)
                    val_solutions.extend(val_solutions_iter)

    # put add the final goals and solutions to the validation set
    val_goals = [f"Conj:(hard) {g}" for g in val_goals_formatted]

    val_goals_formatted.extend(final_goals_formatted)
    val_goals.extend(final_goals)
    val_solutions.extend(final_solutions)

    # remove duplicates in case there are any
    val_goals_formatted = list(set(val_goals_formatted))
    val_goals = list(set(val_goals))
    val_solutions = list({tuple(item) for item in val_solutions})

    with open(os.path.join(os.path.dirname(__file__), 'theories', cfg.theory.name + '.p')) as f:
        theory = f.read()

    difficulty_buckets = sorted([list(cfg.difficulty_buckets[i].items())[0]
                                 for i in range(len(cfg.difficulty_buckets))],
                                key=lambda kv: kv[1])

    premises = cfg.theory.premises

    d = peano.PyDerivation()
    d.incorporate(theory)
    proven_conjectures = []
    seen_hindsight_goals = set()
    proofs = []
    model_info = dict()

    continue_dir = cfg.get('continue')
    start_iteration = 0

    if continue_dir is not None:
        os.chdir(continue_dir)
        log.info('Continuing run from %s', continue_dir)
        # Find largest iteration number such that i.pt exists.
        log.info('Starting from iteration %d', start_iteration)
        agent = torch.load(f'model.pt')
        with open('model_info.yaml') as f:
            model_info = yaml.safe_load(f)
        start_iteration = model_info['iteration'] + 1

    if cfg.get('freeze_conjecturer', False):
        log.info('Ablation: Freezing conjecturer.')


    with open('log.jsonl', 'w') as log_file:
        for i in range(start_iteration, cfg.agent.policy.total_iterations):
            context = Context(d, None, [])

            # Dump current agent.
            buff = io.BytesIO()
            torch.save(agent, buff)
            agent_dump = buff.getvalue()

            # Check if and how many conjectures out of the final goal set could be proven by current policy
            student_results_final = prove_conjectures(agent_dump, final_goals_formatted, theory, premises)
            success_logprobs_final = get_log_probs(student_results_final, i)
            log.info('Final goals proven: %d out of %d', len(success_logprobs_final), len(final_goals))
            final_goals_proven = len(success_logprobs_final)

            # terminate the learning loop if all final goals are proven
            if len(success_logprobs_final) == len(final_goals):
                final_results = []
                for srf in student_results_final:
                    lines = [l["str"] for l in srf.extracted_examples]
                    final_results.append({"theorem": srf.problem, "proof": lines})
                # write final goals and their proofs to a file
                json.dump(final_results, open('final_goals_proofs.json', 'w'))
                # end the training loop
                log.info('All final goals proven - stopping learning loop...')
                mle_log.update({'num_iterations': i},
                           {'final_goals_proven': final_goals_proven})
                break

            # Check if and how many conjectures out of the val goal set could be proven by current policy
            student_results_val = prove_conjectures(agent_dump, val_goals_formatted, theory, premises)
            success_logprobs_val = get_log_probs(student_results_val, i)
            log.info('Val goals proven: %d out of %d', len(success_logprobs_val), len(val_goals))
            val_goals_proven = len(success_logprobs_val)

            # terminate the learning loop if all final goals are proven
            val_results = []
            for srv in student_results_val:
                lines = [l["str"] for l in srv.extracted_examples]
                val_results.append({"theorem": srv.problem, "proof": lines})
            # write final goals and their proofs to a file
            json.dump(val_results, open(f'val_goals_proofs_{i}.json', 'w'))
            # end the training loop

            # 1- Run conjecturing model to obtain N conjectures.
            log.info('Iteration #%d: making conjectures...', i)

            progress_bar = tqdm(total=cfg.n_conjectures)

            conjectures = []

            while len(conjectures) < cfg.n_conjectures:
                proposal = sample_conjecture(AgentLM(agent, 'Conj:(hard) '), context)

                if proposal and proposal not in conjectures + proven_conjectures:
                    contracted_proposal = d.contract(proposal)
                    if contracted_proposal not in conjectures + proven_conjectures:
                        conjectures.append(contracted_proposal)
                        progress_bar.update(1)

            progress_bar.close()

            # Contract conjectures to make them Peano-parseable.
            conjectured_final_goals = set(conjectures) & set(final_goals_formatted)

            log.info('Done making %d conjectures', len(conjectures))
            log.info('Conjectures: %s', conjectures)
            log.info('Conjectured %d final goals', len(conjectured_final_goals))

            log_file.write(json.dumps({'iteration': i,
                                  'msg': f'It #{i}: posing {len(conjectures)} conjectures.',
                                  'conjectures': conjectures}))
            log_file.write('\n')
            log_file.flush()

            # 2- Try to prove each of the conjectures
            examples = []
            student_results= prove_conjectures(agent_dump, conjectures, theory, premises)

            # 3- Train model on proofs and outcome of conjectures (easy, hard, timeout)
            # 3a- Look at all the success logprobs and compute the easy/hard threhsold.
            success_logprobs = get_log_probs(student_results, i)
            
            ratio_proven = len(success_logprobs)/len(conjectures)
            log.info('%d out of %d conjectures proven. ratio = %f', 
                        len(success_logprobs), len(conjectures), ratio_proven)

            if not success_logprobs:
                log.warning('No solutions found in iteration %d - continuing to next iteration...', i)
                continue

            # Add output of proving final goals to the list of proven conjectures
            student_results.extend(student_results_final)

            thresholds = [np.percentile(success_logprobs, p)
                          for _, p in difficulty_buckets]


            log.debug('Thresholds: %s, min = %f, max = %f',
                        list(zip([k for k, _ in difficulty_buckets], thresholds)),
                        np.min(success_logprobs),
                        np.max(success_logprobs))

            hard_sol_log_probs = [logprob for logprob in success_logprobs if logprob >= thresholds[0]]
            mean_hard_sol_log_prob = np.mean(hard_sol_log_probs) if hard_sol_log_probs else 0
            # 3b- Classify problems into easy/hard.
            for student_result in student_results:
                # Outcome is the name of the first difficulty bucket that is larger than the logprob.
                if student_result.success:
                    outcome = next(k
                                   for i, (k, _) in enumerate(difficulty_buckets)
                                   if (student_result.logprob <= thresholds[i] or
                                       i + 1 == len(difficulty_buckets)))
                else:
                    outcome = FAIL

                if not cfg.get('freeze_conjecturer', False):
                    examples.append(f'Conj:({outcome}) ' + d.elaborate(student_result.problem))

                if student_result.success:
                    proven_conjectures.append(student_result.problem)
                    proofs.append(student_result.proof)

                examples.extend(student_result.extracted_examples)

                if cfg.train_policy_on_hindsight_examples:
                    for h in student_result.hindsight_examples:
                        if h.goal not in seen_hindsight_goals:
                            outcome = next(k
                                           for i, (k, _) in enumerate(difficulty_buckets)
                                           if h.logprob <= thresholds[i] or i + 1 == len(difficulty_buckets))

                            if not cfg.get('freeze_conjecturer', False):
                                examples.append(f'Conj:({outcome}) ' + d.elaborate(student_result.problem))
                            examples.extend(h.examples)
                            seen_hindsight_goals.add(h.goal)

            log_file.write(json.dumps({'iteration': i,
                                  'msg': f'Training on {len(examples)} examples.'}))
            log_file.write('\n')

            # 3c- Train model on conjecturing and proof search examples.
            if i + 1 < cfg.agent.policy.total_iterations:
                print(len(examples), 'accumulated training examples.')
                val_loss = agent.train(examples=examples, final_goals=final_goals, solutions=val_solutions, ratio_proven=ratio_proven, mle_log=mle_log)
                mle_log.update({'num_iterations': i},
                           {'val_loss': val_loss,
                            'val_goals_proven': val_goals_proven,
                            'final_goals_proven': final_goals_proven,
                            'ratio_proven': ratio_proven,
                            'mean_hard_sol_log_probs': mean_hard_sol_log_prob},
                            extra_obj={'conjectured_final_goals': conjectured_final_goals})

            mle_log.save()

            save_json(examples, f'examples_{i}.json')
            torch.save(agent, "model.pt")
            model_info['iteration'] = i
            with open('model_info.yaml', 'w') as f:
                yaml.dump(model_info, f)


def prove_conjectures(agent_dump, conjectures, theory, premises):
    tasks = []
    log.info('Submitting tasks...')
    for conjecture in tqdm(conjectures, miniters=1):
        tasks.append(submit_task(
            agent_dump,
            worker.BackgroundTheory(theory, premises),
            conjecture))

    student_results = []

    log.info('Collecting %d results from workers.', len(tasks))

    for task in tqdm(tasks, miniters=1):
        student_result = get_task_result(task)

        if student_result.error:
            log.error('Error in prover process!')
            log.error(student_result.error)
            continue

        student_results.append(student_result)
    return student_results


def get_log_probs(student_results, i):

    success_logprobs = []

    for student_result in student_results:
        if student_result.success:
            success_logprobs.append(student_result.logprob)

    return success_logprobs



@hydra.main(version_base="1.2", config_path="config", config_name="bootstrap")
def main(cfg: DictConfig):
    log.info('Running from: %s', os.getcwd())
    
    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    mle_log = setup_mle_logger(cfg)

    if cfg.task == 'teacher':
        asyncio.run(teacher_loop(cfg, mle_log))

if __name__ == '__main__':
    main()
