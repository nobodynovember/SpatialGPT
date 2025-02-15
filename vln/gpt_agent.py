import sys
import numpy as np
from collections import defaultdict
from GPT.one_stage_prompt_manager import OneStagePromptManager
from vln.spatial_expert import SpatialExpert
from .agent_base import BaseAgent
from GPT.api import gpt_infer
import json


class GPTNavAgent(BaseAgent):
    env_actions = {
        'left': (0, -1, 0),  # left
        'right': (0, 1, 0),  # right
        'up': (0, 0, 1),  # up
        'down': (0, 0, -1),  # down
        'forward': (1, 0, 0),  # forward
        '<end>': (0, 0, 0),  # <end>
        '<start>': (0, 0, 0),  # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0):
        super().__init__(env)
        self.args = args

        self._build_prompt_manager()
        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)
        self.spatialexpert = SpatialExpert(self.args, self.env, self.prompt_manager) 

    
    def _build_prompt_manager(self):
        self.prompt_manager = OneStagePromptManager(self.args)
        print('Model version:', self.args.llm)

    def make_equiv_action(self, a_t, obs, traj=None):

        def take_action(i, name):
            if type(name) is int:       # Go to the next viewpoint
                self.env.env.sims[i].makeAction([name], [0], [0])
            else:                       # Adjust
                self.env.env.sims[i].makeAction(*self.env_actions[name])

        for i, ob in enumerate(obs):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = ob['candidate'][action]
                src_point = ob['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12  # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, 'down')
                    src_level -= 1
                while self.env.env.sims[i].getState()[0].viewIndex != trg_point:    # Turn right until the target
                    take_action(i, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[i].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, select_candidate['idx']) # j+1: idx for navigable location

                state = self.env.env.sims[i].getState()[0]
                if traj is not None:
                    traj[i]['path'].append([state.location.viewpointId])

    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
            self.spatialexpert.reset()  
            print('-------------------- Rollout Reset --------------------')
        else:
            obs = self.env._get_obs()
            print('-------------------- Rollout Continue --------------------') 
        
        batch_size = len(obs)
        
        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
            'a_t': {},
        } for ob in obs]

        if traj[0]['instr_id'] in self.results:
            return [None]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        previous_angle = [{'heading': ob['heading'],
                               'elevation': ob['elevation']} for ob in obs]

        self.prompt_manager.history = ['' for _ in range(self.args.batch_size)]
        self.prompt_manager.nodes_list = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.node_imgs = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.graph = [{} for _ in range(self.args.batch_size)]
        self.prompt_manager.trajectory = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.planning = [["Navigation has just started, with no planning yet."] for _ in range(self.args.batch_size)]
        # print('batch_size:', batch_size)
        sys.stdout.flush()
        self.spatialexpert.spatial_extract_instruction(obs) 
        for t in range(self.args.max_action_len):
            if t == self.args.max_action_len:
                break
                           
            # record visual observations in spatial knowledge graph
            normal=self.spatialexpert.save_spatial_observation(obs, t, self.env) 
            if normal==1:
                # print('[calling spatial_reasoning]')
                s_t=self.spatialexpert.spatial_reasoning(obs, t) 
            
            # Prepare environment action
            cpu_a_t = []
            cpu_a_t.append(0)
            
            # compare next node with the ground truth
            m_index = None
            s_index = None
            g_index = None
            m_action = None
            s_action = None
            g_action = None
            # mapcandidate=obs[0]['candidate'][cpu_a_t[0]]
            scan, gt_traj = self.env.gt_trajs[obs[0]['instr_id']]
            print('Grounded Path:', gt_traj)
            # print('t:', t)
            
            if self.spatialexpert.stop_flag==True:
                print('[SpatialGPT] Stopped')
                #switch for run spatial expert
                break 
            
          
            if t+1 <= len(gt_traj)-1:
                g_index=gt_traj[t+1]
     
            # for candidate in obs[0]['candidate']:
            for i in range(len(obs[0]['candidate'])):
                candidate = obs[0]['candidate'][i]
                candidate_index=candidate['pointId']
                  
                if candidate['viewpointId'] == s_t['viewpointId']:
                    s_index = candidate['viewpointId']
                    s_action = i
                    # print('SpatialGPT index:', s_index)
                    # print('SpatialGPT action:', s_action)
            
                # if candidate['viewpointId'] == g_index: # switch for run spatial expert
                #     g_action = i
                for j in range(len(gt_traj)):
                    if candidate['viewpointId'] == gt_traj[j]:
                        g_action = i
                        break
            # print('[SpatialGPT] Select Direction:', s_index)
            # print('[Grounded] Select Direction:', g_index)
            
            cpu_a_t[0] = s_action
            #cpu_a_t[0] = g_action

            self.make_equiv_action(cpu_a_t, obs, traj)
            obs = self.env._get_obs()

            previous_angle = [{'heading': ob['heading'],
                               'elevation': ob['elevation']} for ob in obs]
            


        return traj
