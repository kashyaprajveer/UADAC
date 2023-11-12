import numpy as np

class OfflineEnv(object):
    
    def __init__(self, users_dict, users_history_lens, movies_id_to_movies, state_size, fix_user_id=None):

        self.users_dict = users_dict
        self.users_history_lens = users_history_lens
        self.items_id_to_name = movies_id_to_movies
        
        self.state_size = state_size
        self.available_users = self._generate_available_users()

        self.fix_user_id = fix_user_id

        self.user = fix_user_id if fix_user_id else np.random.choice(self.available_users) # choose user randomly
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}  #dict {movie as key and rating as value}.
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]] # filter using userID and get the first 10 Movie_ID
        self.done = False
        self.recommended_items = set(self.items) # Movie-list containing last 10(state_size) movies that user has rated 
        self.done_count = 3000
        
    def _generate_available_users(self): # Filter USER and keep user with more than 10 review history.
        available_users = []
        for i, length in zip(self.users_dict.keys(), self.users_history_lens):
            if length > self.state_size:
                available_users.append(i)  #USERD_ID
        return available_users
    
    def reset(self):
        self.user = self.fix_user_id if self.fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]} # Dictionary of {movie_id:rating} for some random user.
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]] # first 10(state size) movies rated.
        self.done = False
        self.recommended_items = set(self.items) # Already recommended
        return self.user, self.items, self.done
     
    # Called from recommender.py using ->  next_items_ids, reward, done, _ = self.env.step(recommended_item, top_k=top_k)    
    def step(self, action, top_k=False):

        reward = -0.5
        
        if top_k:
            correctly_recommended = []
            rewards = []
            for act in action:
                if act in self.user_items.keys() and act not in self.recommended_items:
                    correctly_recommended.append(act)
                    rewards.append(self.user_items[act] - 3)
                else:
                    rewards.append(-0.5)
                self.recommended_items.add(act)
            if max(rewards) > 0:
                self.items = self.items[len(correctly_recommended):] + correctly_recommended
            reward = rewards

        else:
            # If action is in the user's extended rated list(so that we machine can understand his interest) and is not yet recommended
            if action in self.user_items.keys() and action not in self.recommended_items:
                # action is Movie_ID
                # self.user_items.keys() is dictionary with Movie_ID as keys and contains ratings as values
                # self.recommended_items is list of items recommended(but how? -> By selecting first 10 movies he rated)
                
                
                reward = self.user_items[action] -3  # reward which is movie rating - 3
            if reward > 0:
                # concatinate [movie_ID] with list of last 9 Movie_ID now item will increase by 1.
                # Hence the list will still be 10 in size
                self.items = self.items[1:] + [action]
                
            self.recommended_items.add(action)

        # When to stop episodic iteration   
        if len(self.recommended_items) > self.done_count or len(self.recommended_items) >= self.users_history_lens[self.user-1]:
            self.done = True
            
        return self.items, reward, self.done, self.recommended_items

    def get_items_names(self, items_ids):
        items_names = []
        for id in items_ids:
            try:
                items_names.append(self.items_id_to_name[str(id)])
            except:
                items_names.append(list(['Not in list']))
        return items_names
