import re
import math
import os

class OneStagePromptManager(object):
    def __init__(self, args):

        self.args = args
        self.history  = ['' for _ in range(self.args.batch_size)]
        self.nodes_list = [[] for _ in range(self.args.batch_size)]
        self.node_imgs = [[] for _ in range(self.args.batch_size)]
        self.graph  = [{} for _ in range(self.args.batch_size)]
        self.trajectory = [[] for _ in range(self.args.batch_size)]
        self.planning = [["Navigation has just started, with no planning yet."] for _ in range(self.args.batch_size)]

    def get_action_concept(self, rel_heading, rel_elevation):
        if rel_elevation > 0:
            action_text = 'go up'
        elif rel_elevation < 0:
            action_text = 'go down'
        else:
            if rel_heading < 0:
                if rel_heading >= -math.pi / 2:
                    action_text = 'turn left'
                elif rel_heading < -math.pi / 2 and rel_heading > -math.pi * 3 / 2:
                    action_text = 'turn around'
                else:
                    action_text = 'turn right'
            elif rel_heading > 0:
                if rel_heading <= math.pi / 2:
                    action_text = 'turn right'
                elif rel_heading > math.pi / 2 and rel_heading < math.pi * 3 / 2:
                    action_text = 'turn around'
                else:
                    action_text = 'turn left'
            elif rel_heading == 0:
                action_text = 'go forward'

        return action_text

    def make_action_prompt(self, obs, previous_angle):

        nodes_list, graph, trajectory, node_imgs = self.nodes_list, self.graph, self.trajectory, self.node_imgs

        batch_view_lens, batch_cand_vpids = [], []
        batch_cand_index = []
        batch_action_prompts = []

        for i, ob in enumerate(obs):
            cand_vpids = []
            cand_index = []
            action_prompts = []

            if ob['viewpoint'] not in nodes_list[i]:
                # update nodes list (place 0)
                nodes_list[i].append(ob['viewpoint'])
                node_imgs[i].append(None)

            # update trajectory
            trajectory[i].append(ob['viewpoint'])

            # cand views
            for j, cc in enumerate(ob['candidate']):

                cand_vpids.append(cc['viewpointId'])
                cand_index.append(cc['pointId'])
                direction = self.get_action_concept(cc['absolute_heading'] - previous_angle[i]['heading'],
                                                          cc['absolute_elevation'] - 0)

                if cc['viewpointId'] not in nodes_list[i]:
                    nodes_list[i].append(cc['viewpointId'])
                    node_imgs[i].append(cc['image'])
                    node_index = nodes_list[i].index(cc['viewpointId'])
                else:
                    node_index = nodes_list[i].index(cc['viewpointId'])
                    node_imgs[i][node_index] = cc['image']

                action_text = direction + f" to Place {node_index} which is corresponding to Image {node_index}"
                action_prompts.append(action_text)

            batch_cand_index.append(cand_index)
            batch_cand_vpids.append(cand_vpids)
            batch_action_prompts.append(action_prompts)

            # update graph
            if ob['viewpoint'] not in graph[i].keys():
                graph[i][ob['viewpoint']] = cand_vpids

        return {
            'cand_vpids': batch_cand_vpids,
            'cand_index':batch_cand_index,
            'action_prompts': batch_action_prompts,
        }

    def make_action_options(self, cand_inputs, t):
        action_options_batch = []  # complete action options
        only_options_batch = []  # only option labels
        batch_action_prompts = cand_inputs["action_prompts"]
        batch_size = len(batch_action_prompts)

        for i in range(batch_size):
            action_prompts = batch_action_prompts[i]
            if bool(self.args.stop_after):
                if t >= self.args.stop_after:
                    action_prompts = ['stop'] + action_prompts

            full_action_options = [chr(j + 65)+'. '+action_prompts[j] for j in range(len(action_prompts))]
            only_options = [chr(j + 65) for j in range(len(action_prompts))]
            action_options_batch.append(full_action_options)
            only_options_batch.append(only_options)

        return action_options_batch, only_options_batch

    def make_history(self, a_t, nav_input, t):
        batch_size = len(a_t)
        for i in range(batch_size):
            nav_input["only_actions"][i] = ['stop'] + nav_input["only_actions"][i]
            last_action = nav_input["only_actions"][i][a_t[i]]
            if t == 0:
                self.history[i] += f"""step {str(t)}: {last_action}"""
            else:
                self.history[i] += f""", step {str(t)}: {last_action}"""

    def make_map_prompt(self, i):
        # graph-related text
        trajectory = self.trajectory[i]
        nodes_list = self.nodes_list[i]
        graph = self.graph[i]

        no_dup_nodes = []
        trajectory_text = 'Place'
        graph_text = ''

        candidate_nodes = graph[trajectory[-1]]

        # trajectory and map connectivity
        for node in trajectory:
            node_index = nodes_list.index(node)
            trajectory_text += f""" {node_index}"""

            if node not in no_dup_nodes:
                no_dup_nodes.append(node)

                adj_text = ''
                adjacent_nodes = graph[node]
                for adj_node in adjacent_nodes:
                    adj_index = nodes_list.index(adj_node)
                    adj_text += f""" {adj_index},"""

                graph_text += f"""\nPlace {node_index} is connected with Places{adj_text}"""[:-1]

        # ghost nodes info
        graph_supp_text = ''
        supp_exist = None
        for node_index, node in enumerate(nodes_list):

            if node in trajectory or node in candidate_nodes:
                continue
            supp_exist = True
            graph_supp_text += f"""\nPlace {node_index}, which is corresponding to Image {node_index}"""

        if supp_exist is None:
            graph_supp_text = """Nothing yet."""

        return trajectory_text, graph_text, graph_supp_text
    
    def make_synchronize_bydcel_prompt(self, step_instru, prev_instru, dcel):
        prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. You are provided below with (1) a sequence of stepwise navigation instructions. (2) the previous step instruction that the agent was performing, indicating the movement from the previous observation point to the current observation point.. (3) The landmarks viewed by the agent, grouped according to their relative direction to the agent.    Your task is to judge whether the agent has completed the previous step instruction (i.e. has it already {prev_instru})?  and provide a brief reason. Your response should be in JSON format as follows:    "completed_status": " ", "brief_reason":" " . Note: (1) completed_status field should be filled with "yes" or "no" based on your judgment on the previous step instruction's completion status. (2) The viewed landmarks' relative direction to the agent is categorized into four areas: front, back, left, and right. One area may contains multiple landmarks separated by comma. (3) If the previous step instruction involves only a turning action without any related landmark, there is no need for judgment, simply fill the completed_status field with "yes". (4) If the previous step instruction's related landmark is mentioned again in the subsequent step instruction, there is no need for judgement, simply fill the completed_status field with "yes". (5) For normal indoor object landmarks in the previous step, such as tables or doors, their appearance in the agent's front area indicates that the agent has not yet completed the previous step of "passing" or "going through" the landmark. Conversely, their appearance in the agent's back area indicates that the agent has already completed the previous step of "passing" or "going through" the landmark. For large-field landmarks from the previous step, where the agent can be inside (e.g., a room or path), if the landmark appears in multiple areas around the agent, it indicates that the agent has completed the previous step of "entering" the large-field landmark but has not yet completed the step of "exiting" the large-field landmark. If such a large-field landmark from the previous step appears only in the back area of the agent, it signifies that the agent has completed the previous step of "exiting" this large-field landmark. """
        prompt_user = f""" Stepwise Navigation Instruction:{step_instru}. Previous Step Instruction:{prev_instru}. Landmarks' Relative Direction:{dcel}"""
        return prompt_system, prompt_user

    def make_gpt_stop_reason_prompt(self, ob, instruction, navigable_images):
        prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' is provided below, offering navigation stop guidance. Your task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If you cannot see the detination, fill the stop_distance field with "-1". """ 
        prompt_user = f""" Navigation Instruction:{instruction}"""
        return prompt_system, prompt_user

    def make_gpt_breadth_reason_prompt(self, ob, instruction, orientation, navigable_images, progress):
        left_1 = (orientation - 2) % 12
        left_2 = (orientation - 3) % 12
        right_1 = (orientation + 2) % 12
        right_2 = (orientation + 3) % 12
        forward_1 = (orientation + 1) % 12
        forward_2 = (orientation - 1) % 12
        back_1 = (orientation + 5) % 12
        back_2 = (orientation + 6) % 12
        back_3 = (orientation + 7) % 12
        prompt_system = f"""You are a Vision-Language Navigation (VLN) agent navigating in the real world. You are positioned at an observation point within an indoor environment. The surrounding environment is represented by provided 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees.  Each image, with its image_id incremented by 1, corresponds to a 30-degree rightward turn from the previous image. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images.  The image with image_id {orientation}, or {forward_1}, or {forward_2} corresponds to your 'forward', or 'ahead', or 'straight' direction; image_ids {left_1}, or {left_2} correspond to your 'left' direction; image_ids {right_1}, or {right_2} correspond to your 'right' direction; image_ids {back_1}, or {back_2}, or {back_3} correspond to your 'back' direction.   In these 12 images, only the images with indices of {navigable_images} contain navigable path. Additionally, a 'navigation instruction' and 'estimated progress' are provided below. The navigation instruction offers step-by-step guidance for navigation, while the estimated progress, extracted from the given navigation instruction, represents the agent's current instruction execution progress. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to align the relevant visual information from the provided environment images with the navigation instruction to find a path to the destination and stop.  Please use the multimodal embedding-based method, evaluate the alignment between each direction's image and the full sentence of given navigation instruction by encoding both into a shared feature space with the CLIP model and calculating their cosine similarity. During the similarity calculation process, you should ignore the steps preceding the given "estimated progress" and focus only on the steps from the "estimated progress" onward in the navigation instructions.   Based on the resulted similarity scores for all directions, you must select an provided image with the highest similarity score and generate a path plan. The path plan should include a following path plan description and a comparison of all the possible images with navigable paths. Your response should be in JSON format as follows: 'stop_distance':' ', 'path_plan':' ', 'selected_image':' ', 'backup_direction_list':' ', 'score_list': a list of key-value below: 'image_id':' ', 'similarity_score':' '. Notes: 1) Use the number of image ID to fill the selected_image field. 2) backup_direction_list should include a list of all image IDs mentioned in the path plan. 3) The score_list should include 12 directional image items in which each item contains its image_id and its resulted similarity score. 4) If you can already see the destination described in the navigation instruction in any image, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If you cannot see the detination in any image, fill the stop_distance field with "-1". """ 
        prompt_user = f""" Navigation Instruction:{instruction}. Estimated Progress:{progress}"""
        return prompt_system, prompt_user

    def make_spatial_extract_instruction_prompt(self, obs):
        prompt_system = f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. Below is  a Vision-Language Navigation (VLN) task instruction providing step-by-step detailed navigation guidance. Your have two task. The first task is to split the instruction into a sequence of stepwise actions along the time. Each verb should correspond to one stepwise action. The second task is to judge whether the whole navigation includes any elevation-decreasing movement like go downstairs etc. If such movements are included, infer the steps corresponding to those elevation-decreasing movements. The results should be output as JSON in the following struture: 'elevation_decrease':' ', 'step_actions': a list of key-value below: 'action_name':' ', 'landmarks': a nested key-value list in which each item include (landmark_name) and (relative_spatial_area) fields.  Notes: (1)The elevation_decrease field should be filled with a list of step indices (using the stepwise index in the first task's result) corresponding to the steps involving elevation-decreasing movements. If no such elevation-decreasing movements occur during the entire navigation process, the list should be left empty.  (2) If an action refers to no landmark, e.g. "turn left", the 'landmarks' list should be empty list.  (3) An action may refer to multiple landmarks and their relative spatial areas which construct the pair list in the json output. (4) From a spatial perspective, the relative_spatial_area should be mapped to one of four categories: includes: front, back, left, right. (5) For each stepwise action, fill its action_name field with the exact description of that step from the given instruction.  """
        prompt_user = f"""Instruction-{obs[0]["instruction"]}"""

        return prompt_system, prompt_user
    
    def make_spatial_search_landmarks_prompt(self, orientation, step_instru, landmarks):
        # prompt_system=f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. You are positioned at an observation point during a navigation. The surrounding environment is provided to you as 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images. However, the entire 360-degree horizontal view of the environment is covered by these 12 images. The image with image_id {orientation} corresponds to agent's current orientation.    You are provided with a final step of navigation instruction to guide the agent to stop close to the destination. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to search the specified landmarks provided below within these 12 images.   The results should be output as JSON in the following structure: 'stop_distance':' ', 'search_result': a list of key-value below: 'image_id':' ', 'result_list': a nested key-value list in which each item includes (landmark_id), (landmark_name), (type), and (presence) fields.  Note: 1) If you can already see the destination described in the navigation instruction in any image, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If the detination is not visible in any image, fill the stop_distance field with "-1".  Your distance estimation is very important for stop decision, must make it very carefully. 2) The 'search_result' list should include 12 items for each image query result, even if no landmarks are included. 3) presence field should be mapped to one of two caterories: yes, no. 4) From a geometric spatial perspective, landmarks are classified into three types for the Type field of JSON output: a) Point: Specific locations or objects, such as doors, windows, stairway starting points, corridor corners, furniture, etc;  b) PolyLine: Connecting paths or boundaries between two locations, such as hallways, doorways, walls, etc.  c) Polygon: Areas covering a certain space, such as living rooms, bedrooms, carpeted areas, etc: """  
        # prompt_system=f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. You are positioned at an observation point during a navigation. The surrounding environment is provided to you as 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images. However, the entire 360-degree horizontal view of the environment is covered by these 12 images. The image with image_id {orientation} corresponds to agent's current orientation.    You are provided with a full navigation instruction to guide the agent to the final destination and stop. You have two tasks. The first task is to compare the surrounding environment with the given instruction to assess execution progress. If the agent reaches the final step of the given instruction, determine whether it has arrived at the final destination based on visual inputs. The second task is to search the specified landmarks provided below within these 12 images.   The results should be output as JSON in the following structure: 'stop_distance':' ', 'search_result': a list of key-value below: 'image_id':' ', 'result_list': a nested key-value list in which each item includes (landmark_id), (landmark_name), (type), and (presence) fields.  Note: 1) If you determine that you have reached the final step of the given instruction and you have arrived at the final destination, estimate the distance in meters between you and the final destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If you determine that you have not yet reached the final step of the instruction or the final destination, fill the stop_distance field with "-1".  Your judgement is crucial for the stop decision, so make it carefully. 2) The 'search_result' list should include 12 items for each image query result, even if no landmarks are included. 3) presence field should be mapped to one of two caterories: yes, no. 4) From a geometric spatial perspective, landmarks are classified into three types for the Type field of JSON output: a) Point: Specific locations or objects, such as doors, windows, stairway starting points, corridor corners, furniture, etc;  b) PolyLine: Connecting paths or boundaries between two locations, such as hallways, doorways, walls, etc.  c) Polygon: Areas covering a certain space, such as living rooms, bedrooms, carpeted areas, etc: """  
        prompt_system=f"""You are a spatial domain expert assisting a Vision-Language Navigation (VLN) agent with spatial reasoning. You are positioned at an observation point during a navigation. The surrounding environment is provided to you as 12 forward-facing images, where the central horizontal angle of each image differs by 30 degrees. Due to the field of view (FOV) being greater than 30 degrees, there will be slight overlaps between adjacent images. However, the entire 360-degree horizontal view of the environment is covered by these 12 images. The image with image_id {orientation} corresponds to agent's current orientation.    You are provided with a final step of navigation instruction to guide the agent to stop close to the destination. You have two tasks. The first task is to compare the surrounding environment with the destination described in the given instruction to make a stop decision.The second task is to search the specified landmarks provided below within these 12 images.   The results should be output as JSON in the following structure: 'stop_distance':' ', 'search_result': a list of key-value below: 'image_id':' ', 'result_list': a nested key-value list in which each item includes (landmark_id), (landmark_name), (type), and (presence) fields.  Note: 1) If you can already see the destination described in the navigation instruction in any image, estimate the distance in meters between you and the destination, then fill the stop_distance field with only the numeric value of the estimated distance as a floating-point number.  If the detination is not visible in any image, fill the stop_distance field with "-1".  Your distance estimation is very important for stop decision, must make it very carefully. 2) The 'search_result' list should include 12 items for each image query result, even if no landmarks are included. 3) presence field should be mapped to one of two caterories: yes, no. 4) From a geometric spatial perspective, landmarks are classified into three types for the Type field of JSON output: a) Point: Specific locations or objects, such as doors, windows, stairway starting points, corridor corners, furniture, etc;  b) PolyLine: Connecting paths or boundaries between two locations, such as hallways, doorways, walls, etc.  c) Polygon: Areas covering a certain space, such as living rooms, bedrooms, carpeted areas, etc: """  
        landmarks_str = "["
        for i in range(len(landmarks)):
            name = landmarks[i]
            landmarks_str += f"""{i+1}:{name};"""
        prompt_user = f"""Final Step of Navigation Instruction:{step_instru}. Landmarks(ID:Name)-{landmarks_str}]"""
        # prompt_user = f"""Full Navigation Instruction:{step_instru}. Landmarks(ID:Name)-{landmarks_str}]"""

        return prompt_system, prompt_user
  
    

    def parse_planning(self, nav_output):
        """
        Only supports parsing outputs in the style of GPT-4v.
        Please modify the parsers if the output style is inconsistent.
        """
        batch_size = len(nav_output)
        keyword1 = '\nNew Planning:'
        keyword2 = '\nAction:'
        for i in range(batch_size):
            output = nav_output[i].strip()
            start_index = output.find(keyword1) + len(keyword1)
            end_index = output.find(keyword2)

            if output.find(keyword1) < 0 or start_index < 0 or end_index < 0 or start_index >= end_index:
                planning = "No plans currently."
            else:
                planning = output[start_index:end_index].strip()

            planning = planning.replace('new', 'previous').replace('New', 'Previous')

            self.planning[i].append(planning)

        return planning

    def parse_json_planning(self, json_output):
        try:
            planning = json_output["New Planning"]
        except:
            planning = "No plans currently."

        self.planning[0].append(planning)
        return planning

    def parse_action(self, nav_output, only_options_batch, t):
        """
        Only supports parsing outputs in the style of GPT-4v.
        Please modify the parsers if the output style is inconsistent.
        """
        batch_size = len(nav_output)
        output_batch = []
        output_index_batch = []

        for i in range(batch_size):
            output = nav_output[i].strip()

            pattern = re.compile("Action")  # keyword
            matches = pattern.finditer(output)
            indices = [match.start() for match in matches]
            output = output[indices[-1]:]

            search_result = re.findall(r"Action:\s*([A-M])", output)
            if search_result:
                output = search_result[-1]

                if output in only_options_batch[i]:
                    output_batch.append(output)
                    output_index = only_options_batch[i].index(output)
                    output_index_batch.append(output_index)
                else:
                    output_index = 0
                    output_index_batch.append(output_index)
            else:
                output_index = 0
                output_index_batch.append(output_index)

        if bool(self.args.stop_after):
            if t < self.args.stop_after:
                for i in range(batch_size):
                    output_index_batch[i] = output_index_batch[i] + 1  # add 1 to index (avoid stop within 3 steps)
        return output_index_batch

    def parse_json_action(self, json_output, only_options_batch, t):
        try:
            output = str(json_output["Action"])
            if output in only_options_batch[0]:
                output_index = only_options_batch[0].index(output)
            else:
                output_index = 0

        except:
            output_index = 0

        if bool(self.args.stop_after):
            if t < self.args.stop_after:
                output_index += 1  # add 1 to index (avoid stop within 3 steps)

        output_index_batch = [output_index]
        return output_index_batch
