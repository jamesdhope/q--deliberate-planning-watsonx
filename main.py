import json
from watsonx import expert_model, primary_model
import numpy as np
from sentence_transformers import SentenceTransformer, util

question = "What do I need to do to retire as a millionaire?"

max_states_dropout = 10 # maximum number of states to explore before dropout
top_k = 3 # maxumum number of actions to explore from the current state
semantic_similarity_threshold = 0.6 # semantic similarity of state (based on previous actions) to considered 'visited'
_lambda = 1.0

def extract_actions(generated_text):
    # Split the response into lines and filter out the ones containing actions
    actions = []
    for line in generated_text.split('\n'):
        if line.startswith("Action "):
            action = line.split(": ", 1)[1]
            actions.append(action)
    return actions

def calculate_logprobs(results):
    log_probs = []

    if isinstance(results, str):
        results = json.loads(results)

    for chunk in results['results']:
        # keep track of logprob in generated tokens
        if 'generated_tokens' in chunk and isinstance(chunk['generated_tokens'], list):
            for token in chunk['generated_tokens']:
                if 'logprob' in token:
                    log_probs.append(token['logprob'])
    return log_probs

def calculate_reward(log_probs):
    '''maintains log scale. returns value between 0 and infinity'''
    return -np.mean(log_probs)

def get_top_k_actions(state,question):
    '''
    return the top-k actions for the state
    '''
    actions_in_state = [action for action in state.get('actions', []) if action]
    policy = f'''Plan {top_k} distinct and different actions that you can undertake as a language model to better answer the question. Do not repeat any of the actions if there are any actions in this list of previous actions: {actions_in_state}. Provide your best response to the question based on your understanding. Answer in the format:

    Action 1: <Best Action to take>
    Action 2: <Next best Action to take>
    Action 3: <Next best Action to take>
    ...
    Action n: <Next best Action to take>

    Answer: <your best answer>
    '''

    # generate actions under the policy, pi theta
    results = primary_model.generate(prompt=question+policy)

    generated_text = results['results'][0]['generated_text']
    actions = extract_actions(generated_text)
    actions_with_rewards = []

    # use a stronger LLM to compute the reward for each action-trajectory
    for action in actions:
        eval_policy = f'''
        Use the following action and any subsequent actions of your own choosing to answer the question. 
        The action is {action}
        The question is {question}

        Come back to me with a final response to the question having completed the action.
        '''
        results = expert_model.generate(prompt=eval_policy)
        log_probs = calculate_logprobs(results)
        reward = calculate_reward(log_probs)
        
        #map actions to their reward values
        actions_with_rewards.append({'action':action,'q_value':reward})
        actions_with_rewards.sort(key=lambda x: x['q_value'], reverse=True)

    return actions_with_rewards

def s_prime_in_closed_list(state,closed_list):
    '''
    returns semantic simularity score of state_prime to the states in the closed_list based on the actions associated with each state
    '''
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Extract actions from closed_list and filter out empty strings
    actions_in_closed_list = [action for entry in closed_list for action in entry.get('actions', []) if action]
    actions_in_state = [action for action in state.get('actions', []) if action]

    # closed list may be empty during first iteration so return 0 and force state to be added
    if not actions_in_closed_list or not actions_in_state:
        return 0  
    
    embeddings_list1 = model.encode(actions_in_closed_list)
    embeddings_list2 = model.encode(actions_in_state)

    similarities = []

    for emb1 in embeddings_list1:
        for emb2 in embeddings_list2:
            cos_sim = util.pytorch_cos_sim(emb1, emb2)
            similarities.append(cos_sim)

    # how similar is the state based on the semantic similarity of it's actions to other states
    simularity_score = np.mean(similarities)    
    return simularity_score

# Initialize the open and closed lists
open_list = []
closed_list = []  # visited states

inital_state = {
    'value': question, 
    'f_value': 0,
    'actions': []
    }

# Append the initial state q to the open list
open_list.append(inital_state)

no_states = 0
no_actions = 0

print(f'Running Q* with lamda: {_lambda}, max_states_dropout: {max_states_dropout}, top_k_actions: {top_k}, semantic_similarity_threshold: {semantic_similarity_threshold}\n')

while len(open_list) != 0 and no_states < max_states_dropout:

    no_states+=1
    print(f'\nopen list {open_list}\n')

    # SELECT the STATE S in the open list with the highest f-value
    state = max(open_list, key=lambda state: state['f_value']) 
    print(f'Selected State: {state}')

    # REMOVE the selected STATE S from the open list
    open_list.remove(state)
    
    # APPEND the selected STATE S to the closed list
    closed_list.append(state)
    
    # IF the STATE S is a terminal state then extract and return the answer, otherwise continue
    # TODO: if is_terminal(state):  # Assuming each state has an is_terminal method
    #    return extract_answer(state)  # Define extract_answer to get the solution from the terminal state
    
    # FOR EACH ACTION A in the TOP-K ACTIONS generated by the LLM for the STATE S and evaluated by Q*
    top_k_actions = get_top_k_actions(state,question)  
    print(f'Expanding Actions: {top_k_actions}\n')

    for action in top_k_actions: # Expand the Open List / States to Explore
        
        no_actions+=1
        
        # state_prime is the previous state and the action
        state_prime =  {'f_value': state['f_value'] + _lambda*action['q_value'],'actions': state['actions'] + [action['action']]}

        # IF the STATE S-PRIME is not visited then APPEND the STATE S-PRIME to the open list
        if s_prime_in_closed_list(state_prime,closed_list) < semantic_similarity_threshold:
            open_list.append({'f_value': state['f_value'] + _lambda*action['q_value'],'actions': state['actions'] + [action['action']]}) 

state = max(closed_list, key=lambda state: state['f_value'])
actions_in_state = [action for action in state.get('actions', []) if action]

print('State with the highest utility has the following actions:')
for idx, item in enumerate(actions_in_state, start=1):
    print(f"{idx}. {item}")

print(f"\nStates Visited: {no_states}, Actions Considered: {no_actions}")