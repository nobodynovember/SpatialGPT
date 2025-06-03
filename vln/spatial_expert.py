import os
import logging
import sys
import math
import numpy as np
import networkx as nx
import json
import re
import torch
# import cv2
from GPT.api import gpt_infer

class SpatialExpert:
    def __init__(self, args, env, prompt_manager):
        self.args = args
        self.env = env
        self.prompt_manager = prompt_manager
        self.actionlist = []
        self.current_node = None
        self.current_instru_step = 0
        self.current_viewIndex = -1
        self.SpatialKnowledgeGraph = nx.Graph() 
        self.Trajectory = [] 
        self.intersections = [] 
        self.stopping = False
        self.stop_flag=False
        self.extracted_instruction=None
        self.dead_end=0
        self.check_down_elevation=False
        self.num_backtrack=0
        self.num_reflect=0
        self.down_elevation_steps=[]
        self.stop_distance=-1
        self.frontier_flag=False
        self.last_action=False
        self.stop_type=0

    def reset(self):
        if self.SpatialKnowledgeGraph is not None:
            self.SpatialKnowledgeGraph.clear()
        self.Trajectory = []
        self.intersections = []
        self.current_node = None
        self.current_instru_step = 0
        self.current_viewIndex = -1
        self.actionlist = []
        self.stopping = False
        self.stop_flag=False
        self.extracted_instruction=None
        self.check_down_elevation=False
        self.down_elevation_steps=[]
        self.stop_distance=-1
        self.frontier_flag=False
        self.last_action=False
        self.stop_type=0

    def spatial_extract_instruction(self, obs):
        # parse the instruction into a sequence of actions and landmarks with spatial area
        prompt_system, prompt_user=self.prompt_manager.make_spatial_extract_instruction_prompt(obs)
        #print('extract instruction prompt:', prompt_system, prompt_user)
        if self.args.llm == 'gpt-4o' and self.args.response_format == 'json':
                nav_output, tokens = gpt_infer(prompt_system, prompt_user, [],
                                               self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})
                # print("Debug: nav_output content before parsing:", nav_output)
                json_output = json.loads(nav_output)
                self.actionlist = json_output['step_actions']
                try:
                    self.stop_type = int(json_output['stop_type'])
                except ValueError:
                    self.stop_type = 0
                # self.stop_type=0 #debug
                # print('stop type:', self.stop_type)
                # check if the last action has no related landmarks, remove the action from the list
                if len(self.actionlist)>0:            
                    last_action_withdot=self.actionlist[-1]['action_name'].lower()
                    last_action_name = re.sub(r'[^\w\s]', '', last_action_withdot)
                    # print('last action name:', last_action_name)
                    if last_action_name=='stop' \
                        or last_action_name=='stop there'\
                        or last_action_name=='wait'\
                        or last_action_name=='wait there'\
                        or last_action_name=='wait here':
                        self.actionlist.pop()
                
                print('Navigation Instruction Steps:', self.actionlist)

                    # # if last landmark is stair, append a stop action with the stair landmark
                    # for item in self.actionlist[-1]['landmarks']:
                    #     if "stair" in item['landmark_name']:
                    #       self.actionlist.append({'action_type':'Stop', 'action_name': 'Stop', 'landmarks': item['landmarks']})

                if len(json_output['elevation_decrease'])>0:
                    self.check_down_elevation=True
                    self.down_elevation_steps=json_output['elevation_decrease']

    def build_landmark_dcel_area(self, search_result_list, viewIndex):
        front_landmarks=[]
        back_landmarks=[]
        left_landmarks=[]
        right_landmarks=[]
        viewIndex=viewIndex%12
        # print('Agent heading viewIndex:', viewIndex)

        for image_landmarks in search_result_list:
            relative_index=int(image_landmarks['image_id'])-viewIndex
            
            
            if relative_index<0:
                relative_index=relative_index+12

            if 0<=relative_index<=1 or relative_index==11: # front
                for item in image_landmarks['result_list']:
                    if item['presence']=='yes' and item['landmark_name'] not in front_landmarks:
                        front_landmarks.append(item['landmark_name'])
            elif 4<=relative_index<=8: # back, wider for depth reasoning
                for item in image_landmarks['result_list']:
                    if item['presence']=='yes' and item['landmark_name'] not in back_landmarks:
                        back_landmarks.append(item['landmark_name'])
            elif 9<=relative_index <=10: # left
                for item in image_landmarks['result_list']:
                    if item['presence']=='yes' and item['landmark_name'] not in left_landmarks:
                        left_landmarks.append(item['landmark_name'])
            elif 2<= relative_index <= 3: # right, may add more right images
                for item in image_landmarks['result_list']:
                    if item['presence']=='yes' and item['landmark_name'] not in right_landmarks:
                        right_landmarks.append(item['landmark_name'])
        landmark_dcel_area={'front':front_landmarks, 'back':back_landmarks, 'left':left_landmarks, 'right':right_landmarks}
        return landmark_dcel_area
    
    # record visual observations in spatial knowledge graph
    def save_spatial_observation(self, obs, t, env):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)        
        
        if len(obs) > 1:
            print('Error: multiple observations in one step')
        ob = obs[0]
        connect_index=-1  
        path_index=-1     
        print('-----------Move to New Obervation Point-------------')

       
        if self.stopping==True: 
            if bool(self.args.stop_after):
                if t >= self.args.stop_after:
                    # if -0.5<self.prev_distance<0.1: # with additional distance estimation of 0
                    #     self.stopping=False
                    #     print('Additional distance estimation invalid, Continue')
                    # else:
                    self.stop_flag=True
                    print('Close Enough to Destination, Navigation STOP')
                    return 0
                else:
                    print('Stop_after not reached, Continue')
                    self.stopping=False

        # check if the current observation node is intersection with >2 connected observation nodes)
        if t == 0: # no matter start node type, add it as the first path intersection
            self.intersections.append(ob['viewpoint'])

        if len(ob['candidate']) > 2: # intersection
            if t > 0:
                self.intersections.append(ob['viewpoint'])
            connect_index=len(self.intersections) #from 1
            
        elif len(ob['candidate']) == 2: # intermidate node between intersection
            connect_index=0
            
        elif len(ob['candidate']) == 1: # dead end
            if t > 0:
                self.intersections.append(ob['viewpoint']) # dead end is also an path intersection
            connect_index=-1
            
        path_index=len(self.intersections) #from 1
        current_step_landmarks=[]

        j=self.current_instru_step-1
        if t==0: # start node
            j=self.current_instru_step
        # print('[--------Actionlist----------]:', self.actionlist)
        for i in range(j, len(self.actionlist)):  #include current step to last's all landmarks from instruction         
            for landmark in self.actionlist[i]["landmarks"]:
                current_step_landmarks.append(landmark['landmark_name'])
        current_step_landmarks=list(set(current_step_landmarks))
        # print('[-------Current_step---------]:', self.current_instru_step)
        # print('after remove duplicate landmarks,landmarks to check:', current_step_landmarks)
        if t==0: # start node
            orientation=ob['viewIndex']%12
        else:
            orientation=self.current_viewIndex%12
   
       
        final_instru = self.actionlist[len(self.actionlist)-1]['action_name']

        final_instru = final_instru.replace("up the stairs", "up the stairs and stop at the top")
        final_instru = final_instru.replace("down the stairs", "down the stairs and stop at the bottom")
        # print('Final Instruction:', final_instru)
        # print('Full instruction:', full_instru)
        search_result_list=[]
        estimate_distance=-1
        if len(current_step_landmarks) > 0: # if with landmark instructed, search landmarks in the current observation
            prompt_system, prompt_user = self.prompt_manager.make_spatial_search_landmarks_prompt(orientation, final_instru, current_step_landmarks)
            # prompt_system, prompt_user = self.prompt_manager.make_spatial_search_landmarks_prompt(orientation, full_instru, current_step_landmarks)

            # print('search landmarks prompt:', prompt_system, prompt_user)
            image_list = []
            for ix in range(12):
                img_path = os.path.join(env.args.img_root, ob['scan'], ob['viewpoint'], str(ix+12) + '.jpg')
                image_list.append(img_path)               

            if self.args.llm == 'gpt-4o' and self.args.response_format == 'json':
                nav_output, tokens = gpt_infer(prompt_system, prompt_user, image_list,
                                            self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})
                # print("Debug: nav_output content before parsing:", nav_output)                
                json_out=json.loads(nav_output)
                estimate_distance=json_out['stop_distance']
                # print('1st Distance Estimation (Meter):', estimate_distance)
                # print('last step distance:', self.stop_distance)
                
                search_result_list=json_out['search_result']
                # print('search_result_list:', search_result_list)           
        else:
            print('No Related Landmarks for Visual Search')

        
        if(t==0): # start node
            self.current_viewIndex=ob['viewIndex']%12
        dcel_area=self.build_landmark_dcel_area(search_result_list, self.current_viewIndex)
       
        tmp_label=self.current_viewIndex
        print('[GPT Visual Observations for Updating SKG]')

        #add candidate nodes and edges to the spatial knowledge graph
        candidate_list=[]
        for candidate in ob['candidate']:           
            self.SpatialKnowledgeGraph.add_node(str(t)+"-"+candidate['viewpointId'],index=candidate['viewpointId'],
                label=candidate['pointId']%12, prev_label=tmp_label, visited=False, intersection_index=-1, path_index=-1,
                position=candidate['position'], heading=candidate['heading'], elevation=candidate['elevation'],
                search_landmarks=[], landmarks_spatial_area=[], action_step=t,candidates=[], backup=[], instru_step=self.current_instru_step,location_estimation=-1)
            candidate_list.append(candidate)
            print('     Add Candidate Node:', candidate['viewpointId'])
            # print('its image direction:', candidate['pointId'])
            
        if ob['viewpoint'] in self.SpatialKnowledgeGraph.nodes:
            if self.SpatialKnowledgeGraph.nodes[ob['viewpoint']].get('prev_label') is not None:
                tmp_label=self.SpatialKnowledgeGraph.nodes[ob['viewpoint']]['prev_label']
        
        self.SpatialKnowledgeGraph.add_node(ob['viewpoint'],index=ob['viewpoint'],
            label=self.current_viewIndex,prev_label=tmp_label, visited=True, intersection_index=connect_index, path_index=path_index,
            position=ob['position'], heading=ob['heading'], elevation=ob['elevation'], 
            search_landmarks=current_step_landmarks, landmarks_spatial_area=search_result_list, 
            dcel_area=dcel_area, action_step=t, candidates=candidate_list, backup=[], instru_step=self.current_instru_step, location_estimation=float(estimate_distance))
       
        self.current_node = self.SpatialKnowledgeGraph.nodes[ob['viewpoint']]
        print('     Add Observation Node:', ob['viewpoint'])  
        print('     Set Node Property of Agent Orientation:', self.current_viewIndex)
        print('     Set Node Property of VLN-DCEL:', dcel_area)
        # print('its networkx id:', t)
        self.Trajectory.append(self.current_node)
        return 1 
  
    #     return matched_nodelist
    def GPT_front_landmark_aligned(self, instru_step, ob,t): 
        def sort_candidates_elevation(candidate_list):
            sorted_list = sorted(candidate_list, key=lambda x: x['elevation'])
            return sorted_list
        import ast

        import ast

        def parse_numbers(input_data):
            """Convert a string of numbers or a list into a Python list of integers."""
            
            # If input is already a list, return it directly
            if isinstance(input_data, list):
                return [int(x) for x in input_data]  # Ensure all elements are integers
            
            # Convert to string and remove spaces
            input_str = str(input_data).strip()

            try:
                # If input looks like a list (contains [ and ]), use ast.literal_eval
                if input_str.startswith("[") and input_str.endswith("]"):
                    return ast.literal_eval(input_str)
                else:
                    # Otherwise, assume comma-separated numbers and split manually
                    return [int(x.strip()) for x in input_str.split(",")]
            
            except Exception as e:
                print(f"Error parsing input: {e}")
                return []   # Return empty



    
        reason_result=None       
        matched_nodelist=[]
        current_action_name=self.actionlist[instru_step]['action_name']

       
        self.extracted_instruction=ob['instruction']
    #  parse the instruction into a sequence of actions and landmarks with spatial area
        orientation=self.current_viewIndex
        arrival=(self.current_viewIndex+6)%12
        navigable_images=[]
        navigable_images_origin=[]
        navigable_candidates=[]
        for k in range(len(ob['candidate'])):
            candidate = ob['candidate'][k]
            # print('candidate position')
            navigable_images_origin.append(candidate['pointId']%12)
            navigable_candidates.append(candidate)
        navigable_images = list(dict.fromkeys(navigable_images_origin)) # remove duplicates
        # print('navigable_images:', navigable_images)

        if instru_step==len(self.actionlist)-1: # last action
            current_action_name = current_action_name.replace("up the stairs", "up the stairs and stop at the top")
            current_action_name = current_action_name.replace("down the stairs", "down the stairs and stop at the bottom")
            # print('last action name:', current_action_name)

       
        if (instru_step==len(self.actionlist)-1): # last action
            prompt_system, prompt_user=self.prompt_manager.make_gpt_stop_distance_reason_prompt(ob, current_action_name, navigable_images)
        else:
            prompt_system, prompt_user=self.prompt_manager.make_gpt_breadth_distance_reason_prompt(ob, self.extracted_instruction,orientation, navigable_images, current_action_name)


        
        image_list = []
        temp_list=[]
        for ix in range(12):
            img_path = os.path.join(self.env.args.img_root, ob['scan'], ob['viewpoint'], str(ix+12) + '.jpg')
            image_list.append(img_path) 
   
        if self.args.llm == 'gpt-4o' and self.args.response_format == 'json':
            nav_output, tokens = gpt_infer(prompt_system, prompt_user, image_list,
                                        self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})
            reason_result = json.loads(nav_output)
        # print('reason_result:', reason_result)  
        down_reason_result=None
        current_landmark_name=None
        if self.check_down_elevation==True:
            int_down_steps=[int(item)-1 for item in self.down_elevation_steps]
            # print('down elevation steps:', int_down_steps)
            int_down_steps = sorted(int_down_steps)
            if int_down_steps[0] > 0:
                int_down_steps.insert(0, int_down_steps[0] - 1)
            # print('sorted down elevation steps:', int_down_steps)
        if self.check_down_elevation==True and (self.current_instru_step in int_down_steps):
            image_list.clear()
            temp_list.clear()
            for ix in range(12):
                img_path = os.path.join(self.env.args.img_root, ob['scan'], ob['viewpoint'], str(ix) + '.jpg')
                image_list.append(img_path)
           
            if self.args.llm == 'gpt-4o' and self.args.response_format == 'json':
                nav_output, tokens = gpt_infer(prompt_system, prompt_user, image_list,
                                            self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})
                down_reason_result = json.loads(nav_output)

        # print('Agent Orientation:', self.current_viewIndex)
        landmarks=self.actionlist[instru_step]['landmarks']
        if len(landmarks)>0:
            current_landmark_name = landmarks[0]['landmark_name']
        print('[GPT Spatial Alignment Inference]')
        print('      Landmark at Current Step:', current_landmark_name)
        print('      Execute Movement Action:', current_action_name)
        # print('      Navigable_images:', navigable_images)
        print('      Selected Direction:', reason_result['selected_image'])
        print('      Alternative Direction:', reason_result['backup_direction_list'])
        print('      Subsequent Path Plan:', reason_result['path_plan'])      
        if self.check_down_elevation==True and (self.current_instru_step in int_down_steps):
            print('[GPT] Down-Elevation Spatial Alignment Inference-Selected Direction:', down_reason_result['selected_image'])
        matched_nodelist=[]
        down_matched_nodelist=[]
     
        down_highest_candidate=None
        highest_candidate=None
        if self.check_down_elevation==True and (self.current_instru_step in int_down_steps):     
            down_score_list=down_reason_result['score_list']
            down_navigable_scores=[]
         
            # print('navigable_images_origin:', navigable_images_origin)
            for i in range(len(navigable_images_origin)):
                angle=abs(int(down_reason_result['selected_image'])-int(navigable_images_origin[i]))
                gap=min(angle, 12-angle)
                int_navigation=int(navigable_images_origin[i])
                score=float(down_score_list[int_navigation]['similarity_score'])
                elevation=navigable_candidates[i]['elevation']
                down_navigable_scores.append([navigable_candidates[i], gap, score, elevation])
             
            down_navigable_scores=sorted(down_navigable_scores, key=lambda x: (x[1], -x[2], x[3]))            
           
            traj_nodes=[]
            reversed_list = list(reversed(self.Trajectory))
            
            for node in reversed_list:
                traj_nodes.append(node['index'])            
            # print('traj_nodes:', traj_nodes)
            down_not_in_traj = []
            down_in_traj = []
            for item in down_navigable_scores:
                candidate=item[0]
                if candidate['viewpointId'] in traj_nodes:
                    down_in_traj.append(item)
                else:
                    down_not_in_traj.append(item)

            down_navigable_scores.clear()
            down_navigable_scores=down_not_in_traj+down_in_traj
            # a2 = " ".join(item[0]['viewpointId'] for item in down_navigable_scores)
            # print('after down check visited nodes navigable_scores:', a2)
            down_highest_candidate=down_navigable_scores[0][0]           
            down_matched_nodelist.append(down_highest_candidate)
 
        # set backup node for reflect
        backup_list=reason_result['backup_direction_list']
        # print('gpt return backup list:', backup_list)
        
        backup_list=parse_numbers(backup_list)
        # print('after processed backup list:', backup_list)


        # direction_difflist=[]
        # backup_index=-1
        backup_nodelist=[]     
        # sort candidates by gap increasing, for the samp gap, sort by score decreasing
        score_list=reason_result['score_list']
        navigable_scores=[]
        int_backup_list=[int(item) for item in backup_list]
        # print('navigable_images_origin:', navigable_images_origin)
        for i in range(len(navigable_images_origin)):
            angle=abs(int(reason_result['selected_image'])-int(navigable_images_origin[i]))
            gap=min(angle, 12-angle)
            score=float(score_list[navigable_images_origin[i]]['similarity_score'])
            navigable_scores.append([navigable_candidates[i], gap, score])
            # print('append i',i)
            # print('append navigable_candidates:', navigable_candidates[i])
            int_navigation=int(navigable_images_origin[i])

            if int_navigation in int_backup_list:#remember alternative candidates
                # print('int_navigation:', int_navigation)
                backup_nodelist.append(navigable_candidates[i])
        
       
        navigable_scores=sorted(navigable_scores, key=lambda x: (x[1], -x[2]))
        a1 = " ".join(str(item[1]) for item in navigable_scores)
        # print('sort gap:', a1)
        a2 = " ".join(str(item[2]) for item in navigable_scores)
        # print('sort score:', a2)
        traj_nodes=[]
        reversed_list = list(reversed(self.Trajectory))
        
        for node in reversed_list:
            traj_nodes.append(node['index'])        
        # print('traj_nodes:', traj_nodes)
        not_in_traj = []
        in_traj = []
        for item in navigable_scores:
            candidate=item[0]
            if candidate['viewpointId'] in traj_nodes:
                in_traj.append(item)
            else:
                not_in_traj.append(item)
        if self.stopping==False:
            navigable_scores.clear()
            navigable_scores=not_in_traj+in_traj
 
        highest_candidate=navigable_scores[0][0]
        
        matched_nodelist.append(highest_candidate)
 
        if self.check_down_elevation==True and (self.current_instru_step in int_down_steps):
            matched_nodelist=down_matched_nodelist+matched_nodelist
        
        candidate=None
        cirlce_flag=False
        # exclude selected candidate from backup list
        if len(matched_nodelist)>0:
            candidate=matched_nodelist[0]         
            
            for idx, node in enumerate(reversed(self.Trajectory)):
                if idx == 0:
                    continue
                if self.current_node['index'] == node['index']:
                    cirlce_flag=True
                    break
            
            if cirlce_flag==True:
                self.num_backtrack+=1
                print('[Backtrack Planning Initiated]')                       
                if candidate['viewpointId'] in traj_nodes:
                    # print('[Backtrack Plan due to Circle Found]')                       
                    # go to reflect for out of circle
                    reflect_result=self.reflect(ob)
                    if reflect_result is not None:
                        matched_nodelist.clear()
                        matched_nodelist.append(reflect_result)
                    else:
                        print('Frontier Inference due to no alternative node')
                        if self.frontier_flag==True:
                            frontier_result=self.frontier(ob)
                            if frontier_result is not None:
                                matched_nodelist.clear()
                                matched_nodelist.append(frontier_result)
                # else:   
                #     print('Revisited an Accessed Node, Self-Correct to Continue')

            selected_node=matched_nodelist[0]
            if selected_node in backup_nodelist:
                # print('remove selected node from backup:', selected_node['viewpointId'])
                backup_nodelist.remove(selected_node)

        if len(backup_nodelist)>0: 
            # print('add backup_nodelist:', backup_nodelist)
            self.current_node['backup']=backup_nodelist
       

        possible_stop=False
        # if t>=len(self.actionlist)-1:  # after number of actions
        if bool(self.args.stop_after):
            if t >= self.args.stop_after:
                possible_stop=True
            else:
                print('Stop_after not reached, Continue')

   
         
        try:
            self.stop_distance = float(reason_result['stop_distance'])
        except ValueError:
            print("Error: stop_distance is not a valid float value:", reason_result['stop_distance'])
            self.stop_distance = -1.0

     
        estimated_distance=self.current_node['location_estimation']
        print('Additional Estimated Distance (Meter):', estimated_distance)
        if float(self.stop_distance)<-0.5:
            print('Destination Invisible at Distance Estimation')
        else: 
            print('Destination Seen at Distance Estimation (Meter):', self.stop_distance)

     
        if self.stop_distance > -0.5 and possible_stop==True:
            dest_heading_index=int(reason_result['selected_image'])%12
                   
            depth=-1

           
            if self.stop_type==0:  #if object related stop, use depth
                dest_image_path = os.path.join(self.env.args.img_root, ob['scan'], ob['viewpoint'], str(dest_heading_index+12) + '.jpg')
                roi=None

                if current_landmark_name is not None:
                    roi=self.get_ROI_object_depth(dest_image_path, current_landmark_name)

                if roi is not None:
                    print("Top ROI Box:", roi)
                else:
                    print("No object detected.")

                

                #get depth image if detected
                if roi is not None:
                    depth_img = self.get_depth_at_discrete_view(dest_heading_index) 
                    # use median depth value of the image
                    x_min, y_min, x_max, y_max = map(int, roi)
                    roi_depth = depth_img[y_min:y_max, x_min:x_max]
                    valid_depths = roi_depth[roi_depth > 0]

                    if valid_depths.size == 0:
                        print("No valid depth in ROI.")
                    else:
                        depth = np.median(valid_depths)/4000.0
                        if np.isnan(depth) or depth < 0.1 or depth > 10.0:
                            depth = -1
            
                        print(f"Estimated distance to object: {depth:.2f} meters")
            # using depth value if available
            if depth != -1:   #accurate depth value for stop now
                print('!----Using Object Depth:', depth)
                if depth < 4.5:
                    self.stopping=True
            else:  # use GPT estimated distance with higher criteria    
                if float(self.stop_distance)<2.1 or estimated_distance<2.1: 
                    self.stop_flag=True
       
        if self.stopping==True:
            print('Closing to Destination, Agent is Stopping...')  


        return matched_nodelist
    

    def get_ROI_object_depth(self, image_path, object):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from PIL import Image, ImageDraw


        # image_path = "rgb.png"  # Path to your image file
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        else:
            print(f"[INFO] Successfully opened image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        text = object  # Text prompt for the model

        processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        # model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
        model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

        inputs = processor(images=image, text=text, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.image_processor.post_process_object_detection(
            outputs, threshold=0.05, target_sizes=target_sizes
        )[0]
        image_width, image_height = image.size
        print(f"Image size: {image_width} x {image_height}")
        print(f"Detected {len(results['scores'])} objects:", text)

    
        
        scores = results["scores"]
        boxes = results["boxes"]

        if len(scores) == 0:
            return None 

        top_idx = scores.argmax()
        box = boxes[top_idx].tolist()

        # Clamp + round
        x_min = max(0, min(box[0], image_width))
        y_min = max(0, min(box[1], image_height))
        x_max = max(0, min(box[2], image_width))
        y_max = max(0, min(box[3], image_height))
        clamped_box = [round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)]
        print(f"Detected {object} with confidence {round(scores[top_idx].item(), 2)} at location {box}")
    

        return clamped_box

    def get_depth_at_discrete_view(self, heading_index: int, elevation_index: int = 0):
        sim=self.env.env.sims[0]
        # save current state
        
        state = sim.getState()[0]

        # state = sim.getState()[0]
        scan_id = state.scanId
        viewpoint_id = state.location.viewpointId
        current_heading = state.heading
        current_elevation = state.elevation

        # set discrete heading and elevation
        heading_rad = heading_index * math.radians(30)
        elevation_rad = elevation_index * math.radians(30)
        sim.newEpisode([scan_id], [viewpoint_id], [heading_rad], [elevation_rad])
        new_state = sim.getState()[0]

        # get depth image
        depth = np.array(new_state.depth, copy=False)

        # back to original state
        sim.newEpisode([scan_id], [viewpoint_id], [current_heading], [current_elevation])
        return depth

    def frontier(self, ob): # frontier inference
        selected_node=None
        prompt_system, prompt_user=self.prompt_manager.make_frontier_prompt(self.current_instru_stepr, self.current_node, self.SpatialKnowledgeGraph)         
        # print('synchronize prompt:', prompt_system, prompt_user)        
        if self.args.llm == 'gpt-4o' and self.args.response_format == 'json':
                nav_output, tokens = gpt_infer(prompt_system, prompt_user, [],
                                            self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})
                frontier_result = json.loads(nav_output)
                print('[GPT Frontier Inferenece]')
                print('      Selected Frontier:'+ frontier_result['selected_frontier'])
                print('      Reasoning:'+frontier_result['reasoning'])
                selected_node=frontier_result['selected_frontier']
        return selected_node      
    
    def reflect(self, ob):  #decide exploit or explore
        self.num_reflect+=1
        reversed_list = list(reversed(self.Trajectory))
        traj_nodes=[]        
        for node in reversed_list:
            traj_nodes.append(node['index'])
        selected_node=None
        prev_candidate=None
        prev_node=None
        # print('     Checking Alternative Nodes From SKG Current Node..')
        for idx, node in enumerate(reversed(self.Trajectory)):
            if idx == 0:
                continue
            if self.current_node['index'] == node['index']: 
                # check backup memory for exploit       
                for j in range(len(self.current_node['backup'])):
                    backup_candidate=self.current_node['backup'][j]
                    if  backup_candidate in traj_nodes: # exclude the visited nodes
                        continue
                    else:
                        selected_node=backup_candidate
                        print('     Retrieve Alternative Node From SKG:', selected_node['viewpointId'])
                # check previous node
                if idx<len(self.Trajectory)-1:
                    prev_node=list(reversed(self.Trajectory))[idx+1]
                # check unvisited candidates for explore
                if selected_node is None:            
                    for i in range(len(ob['candidate'])):
                        candidate = ob['candidate'][i]
                        if candidate['viewpointId'] in traj_nodes: # exclude the visited nodes
                            if prev_node is not None:
                                if candidate['viewpointId'] == prev_node['index']:
                                    prev_candidate=candidate
                            continue     
                        else:
                            selected_node=candidate               
        # check previous node    
        if selected_node is None:  
            print('Reflect Checked Current Node, Explore to Previous Node')
            selected_node=prev_candidate
        return selected_node

    def breadth_reasoning(self, ob, t):
        selected_node = None            
        matched_nodelist=self.GPT_front_landmark_aligned(self.current_instru_step, ob,t)
        if len(matched_nodelist)>0:
            selected_node = matched_nodelist[0] 
            # step increase if found target
            if self.current_instru_step < len(self.actionlist)-1:
                if self.current_node['intersection_index'] > 0 or t==0: # intersection or start node
                    self.current_instru_step+=1
                    # print('forward step continue+1:', self.current_instru_step)         
        return selected_node
    
    def synchronize_reasoning(self, ob, t):    
        def convert_dcel_area_to_text(dcel_area: dict) -> str:
            
            directions_text = []
            # For each direction (front/back/left/right), list the items found
            for direction in ['front', 'back', 'left', 'right']:
                items = dcel_area[direction]
                if items:
                    items_str = ', '.join(items)
                    directions_text.append(f"{direction} area: {items_str}")
                else:
                    directions_text.append(f"{direction} area: (none)")
                    
            # Join them together into one multiline string
            return " ; ".join(directions_text)

        step_str = ""
        for i in range(len(self.actionlist)):
            action = self.actionlist[i]
            step_str += f"""{i+1}.{action['action_name']};"""

        if self.current_instru_step == 0:
            k=0
        else:
            k=self.current_instru_step-1

        prev_step_str = self.actionlist[k]['action_name']
        
        if t==0:
            return True
        
        # if self.last_action==True:
        #     return True
        
        description_text = convert_dcel_area_to_text(self.current_node['dcel_area'])
        prompt_system, prompt_user=self.prompt_manager.make_synchronize_bydcel_prompt(step_str, prev_step_str, description_text)         
        # print('synchronize prompt:', prompt_system, prompt_user)        
        if self.args.llm == 'gpt-4o' and self.args.response_format == 'json':
                nav_output, tokens = gpt_infer(prompt_system, prompt_user, [],
                                            self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})
                syn_result = json.loads(nav_output)
                print('[GPT Temporal Synchronize Inferenece by VLN-DCEL]')
                print('      Previous Step Completion:'+ syn_result['completed_status'])
                print('      Brief Reason:'+syn_result['brief_reason'])


        if t>0:
            if syn_result['completed_status']=='no':
                landmarks=self.actionlist[self.current_instru_step]['landmarks']
                current_landmark=None
                prev_landmark=None
                if len(landmarks)>0:
                    current_landmark=landmarks[0]['landmark_name']
                landmarks=self.actionlist[self.current_instru_step-1]['landmarks']
                if len(landmarks)>0:
                    prev_landmark=landmarks[0]['landmark_name']
                if current_landmark==prev_landmark and current_landmark is not None:
                    print('Consecutive Steps with the Same Landmark:', current_landmark)
                else:
                    if self.stopping==False:
                        self.current_instru_step-=1    
            # else:
            #     if self.current_instru_step == len(self.actionlist)-1:
            #         # print('Last Step Reached, No Need to Synchronize Any More')     
            #         self.last_action=True
        return True   
    
    
    def continue_straight(self, ob, t):  
        next_node = None
        if (len(ob['candidate']) == 1): # dead-end node
            print('Dead-end Node to the Only Backward Node')
            return ob['candidate'][0]
        
        if len(ob['candidate']) == 2 and t > 0: # intermidate node 
            for candidate in ob['candidate']:
                if candidate['viewpointId'] != self.Trajectory[-2]['index']: #not previous node
                    next_node = candidate
                    break
            print('Intermidate Node to the Only Foward Node')
            return next_node
        
        # select the node closest to the base heading, i.e minimum candidate heading         
        if len(ob['candidate']) > 2 or t == 0: # start node (no-align landmark)
            min_angle=3.141592
            num_candidates=len(ob['candidate'])
            if t > 0: # not start node, previous node out
                num_candidates=num_candidates-1
            # print('straight candidates has:', num_candidates)
            next_node = None
            current_heading=(self.current_viewIndex+0.5)*math.radians(30)
            for candidate in ob['candidate']:   
                # print('candidate absolute heading:', candidate['absolute_heading'])
                gap_angle=abs(candidate['absolute_heading']-current_heading)
                if gap_angle > 3.141592:
                    gap_angle=2*3.141592-gap_angle
                if gap_angle < min_angle:
                    min_angle=gap_angle
                    next_node = candidate
            print('Continue Straight to Node:', next_node)
            # print('its gap angle:', min_angle)            

            return next_node
        print('Error in Continue Straight: No Selected Node')    
        return -1    
    
    # spatial reasoning for the current observation with instruction
    def spatial_reasoning(self, obs, t):
        ob = obs[0]
        selected_node = None       
            
        if t==0: #start node
            # print('Spatial Reasoning On Beginning Node')
            self.extracted_instruction=ob['instruction']
            # if self.depth_reasoning(ob,t)==True: # check progress of instruction
            self.synchronize_reasoning(ob, t)
            selected_node=self.breadth_reasoning(ob, t)
            if selected_node == None:
                print('Alignment: No Node Met Spatial Condition, Go Straight')
                selected_node=self.continue_straight(ob, t) # cannot exploit instruction, go to reflect directly?
                self.current_instru_step+=1
  
        elif self.current_node['intersection_index'] > 0:   # intersection and not the start node
            # print('Spatial Reasoning On Intersection Node')
            # if self.depth_reasoning(ob,t)==True: # check progress of instruction
            if self.synchronize_reasoning(ob, t)==True:
                selected_node=self.breadth_reasoning(ob, t)
                if selected_node is None: # similarity not met, possibility is low
                    print('Alignment: No Node Met Spatial Condition, Go Straight')
                    #selected_node=self.reflect(ob)
                    selected_node=self.continue_straight(ob, t)                    
    

        elif self.current_node['intersection_index'] == 0: # intermidate node between intersection   
            # print('Spatial Reasoning On Intermidate Node')
            selected_node=self.breadth_reasoning(ob, t)
            if selected_node == None:
                print('Alignment: No Node Met Spatial Condition, Go Straight')
                selected_node=self.continue_straight(ob, t)            
        
        elif self.current_node['intersection_index'] == -1: # dead end
            # print('Spatial Reasoning On Dead-end Node')
            selected_node=self.continue_straight(ob, t)            
            self.dead_end+=1
      
        print('[SpatialGPT Select Node:', selected_node['viewpointId'])
        # print('dead end num:', self.dead_end)
        # print('Total Backtrack Times:', self.num_backtrack)
        # print('Total Reflect Times', self.num_reflect)
        print('Latest Trajectory:')
        for node in self.Trajectory:
            print(node['index'])
        if selected_node is not None:
            self.current_viewIndex=int(selected_node['absolute_heading']/ math.radians(30))
            # print('its absolute heading:', selected_node['absolute_heading'])
            # print('its viewIndex:', self.current_viewIndex)
            self.SpatialKnowledgeGraph.add_edge(ob['viewpoint'], selected_node['viewpointId'])

        return selected_node