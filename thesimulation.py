import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from IPython.display import HTML
from tqdm.notebook import tqdm
 
# PARAMETERS CONSOLE
SIZE = 100 # size of the world
ITERS = 100 # number of iterations

N_FISH = 5000 # number of fish
FISH_BETA = 0.6 # momentum factor of fish [0,1] steps of 0.1
FISH_TEMP = 0.5 # softmax temperature of fish [small=min,high=avg] steps of 0.1
FISH_SPACING = 3 # target spacing between fish when schooling in length units so steps of 1
FISH_ATTRACTION = 0.1 # strength of the schooling attraction [0,1] steps of 0.1
RAND_STRENGTH = 0.05 # strength of the random perturbations steps in 0.01
MAX_FISH_VEL = 3 # maximum speed in length units so steps of 1
FISH_VISION = 3 # defines the acuity of the fish vision
ROTATION = True # strength of fish rotation

N_SHARKS = 1 # number of sharks
SHARK_BETA = 0.8 # momentum parameter of sharks
MAX_SHARK_VEL = 6 # maximum velocity of sharks
SHARK_TEMP = 0.001 # softmax temperature of sharks
ATTACK_RADIUS = 6 # visual acuity of sharks
DEATH_RADIUS = 0.5 # radius inside which a shark catches a fish
REST_SPEED = 0.1 # value between [0,1] it is a proportion of MAX_SHARK_VEL
ALIGN_THRESH = 0.999 # alignment needed to start an attack between [0,1]
EATING_LATENCY = 20 # iterations before a shark can start eating again
HUNT_PERIOD = 60 # iterations before a shark can start eating (allows dynamics to settle) 
 
 
# gets colourmap as max of dist_scores for each fish
def get_fish_colourmap(dist_scores):
    # min-max normalise so dist_scores for each shark on same scale
    dist_scores_scaled = dist_scores.clone()
    dist_scores_scaled -= dist_scores_scaled.min(1)[0]
    dist_scores_scaled /= (dist_scores_scaled.max(1)[0] * 2)
    return dist_scores_scaled.max(2)[0].squeeze()

# utils function to return the arrays to plot
def get_ocean(i):
    
    # return live fish with colourmap and sharks
    live_fish = fish[i][:, live_fish_mask[i]]
    curr_fish_colourmap = fish_colourmap[i][live_fish_mask[i]]
    curr_sharks = sharks[i]
    
    # also returns the position of the dead fish
    dead_pos = fish[i, :2, ~live_fish_mask[i]]
    return live_fish, curr_sharks, dead_pos, curr_fish_colourmap

# function that normalises the vectors
def normalise(coordinates, i):
    vecs = coordinates[i, 2:] - coordinates[i, :2]
    norms = torch.sqrt((vecs**2).sum(0, keepdim=True))
    vecs /= norms
    coordinates[i, 2:] = vecs
    
    # ensures that neither fish or sharks overflow from the world
def fix_world_overflow(coordinates, i):
    mask = coordinates[i+1, :2] > SIZE/2
    coordinates[i+1, :2][mask] = -SIZE/2
    coordinates[i+1, :2][mask.flip(0)] *= -1
    mask = coordinates[i+1, :2] < -SIZE/2
    coordinates[i+1, :2][mask] = SIZE/2
    coordinates[i+1, :2][mask.flip(0)] *= -1
    
    # gets the euclidean distance between two sets of point coordinates
# representing fish or shark
def get_distances(coordsA, coordsB, i):
    dist_vecs = coordsA[i+1, :2][:,:,None] - coordsB[i+1, :2][:, None]
    
    # if the distance is more than half a grid, consider the shorter distance from the other side
    dist_vecs[dist_vecs>SIZE/2] -= SIZE
    dist_vecs[dist_vecs<-SIZE/2] += SIZE
    
    # normalise the vectors that point to all the other fish or sharks
    inv_dist_norms = 1 / (torch.sqrt((dist_vecs**2).sum(0, keepdim=True)) + 1e-9)
    return dist_vecs, inv_dist_norms

# creating a custom cmap for fish that increases monotonically
custom_fish_cmap = LinearSegmentedColormap('softmax', 
        {'red':   [(0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)],

         'green': [(0.0,  0.0, 0.0),
                   (1.0,  0.8, 0.8)],

         'blue':  [(0.0,  0.2, 0.2),
                   (0.7,  0.5, 0.5),
                   (1.0,  0.0, 0.0)]
        })
custom_fish_cmap
 
# TILAPIA BEHAVIOR
# returns the forces that act on fish, namely the schooling force and the shark repulsion
def get_fish_forces(i, fs_dist_vecs, fs_inv_dist_norms):
    
    # get the distances between each fish and each other fish
    ff_dist_vecs, ff_inv_dist_norms = get_distances(fish, fish, i)    
    
    # remove the distance from a fish and itself and from a fish and dead fish
    ff_inv_dist_norms *= (1-torch.eye(N_FISH, device=fish.device)[None])
    ff_inv_dist_norms *= live_fish_mask[i][None,None]
    
    # compute a softmax of the most proximal fish
    ff_dist_scores = F.softmax(ff_inv_dist_norms/FISH_TEMP, dim=2)
    
    # use the softmax scores to chose what vector direction to follow and normalise that vector
    attraction_scores = ff_inv_dist_norms * ff_dist_scores
    attraction = (ff_dist_vecs * attraction_scores).sum(2)
    #attraction /= (torch.sqrt(torch.sum((attraction**2), dim=0, keepdim=True)) + 1e-12)
    
    # measure how strongly the fish will be attracted or repelled by another fish by using a tanh function
    # which defines a specific target distance at which a fish should be with respect to the neighbours
    avg_closest_dist = 1 / attraction_scores.sum(2)
    schooling_pull = -torch.tanh((avg_closest_dist-FISH_SPACING)*FISH_ATTRACTION)
    
    # measure the softmax and use it to chose the vector to follow as above
    fs_dist_scores = F.softmax(fs_inv_dist_norms/FISH_TEMP, dim=2)
    repulsion_scores = fs_inv_dist_norms * fs_dist_scores
    repulsion = (fs_dist_vecs * repulsion_scores).sum(2)
    repulsion /= torch.sqrt(torch.sum((repulsion**2), dim=0, keepdim=True))

    
    # avoidance the shark is modelled by a decaying exponential that decays with distance away from the shark
    avg_closest_dist = 1 / repulsion_scores.sum(2)
    repulsion_push = torch.exp(-avg_closest_dist/FISH_VISION)
    
    # adding randomness that depends on the fishes' sense of threat
    randomness = torch.randn(2, N_FISH, device=fish.device)
    
    # add the rotational component
    rotation_left = attraction.flip(0)
    rotation_left[0] *= -1
    rotation_right = attraction.flip(0)
    rotation_right[1] *= -1
    rotation = torch.cat([rotation_left[None], rotation_right[None]], dim=0)
    
    # harmonise fish velocities
    school_rotation = schooling_pull**2 - 1
    #neighbour_vel = (ff_dist_scores * fish[i, 2:, None, :]).sum(2)
    #neighbour_vel /= torch.sqrt((neighbour_vel**2).sum(0, keepdim=True))
    #neigh_vel_gap = (rotation*neighbour_vel[None]).sum(1, keepdim=True)
    #rot_dir = torch.max(neigh_vel_gap, dim=0, keepdim=True)[1]
    #rotation = rotation.gather(0, rot_dir)
    rotation = rotation[0]
    
        
    # the dish drive is composed of following the other fish and avoiding the shark, return it
    rest_movement = schooling_pull*attraction +RAND_STRENGTH*randomness +school_rotation*rotation*ROTATION*FISH_ATTRACTION**(1/2)
    fish_drive = (1-repulsion_push)*rest_movement +repulsion_push*repulsion
    return fish_drive
 
#SHARK BEHVAIOUR
# defines the forces acting on the sharks
def get_shark_forces(i, fs_dist_vecs, fs_inv_dist_norms):
    
    # remove dead fish from the computation because they do not have to be followed anymore
    fs_inv_dist_norms *= live_fish_mask[i][None,:,None]
    
    # use softmax to measure the salience of each fish direction and follow a combination of the closest
    dist_scores = F.softmax(fs_inv_dist_norms/SHARK_TEMP, dim=1)
    hunting_scores = fs_inv_dist_norms * dist_scores
    hunting = (fs_dist_vecs * hunting_scores).sum(1)
    hunting /= torch.sqrt((hunting**2).sum(0, keepdim=True))
    
    # distance of closest density
    avg_closest_dist = 1 / hunting_scores.sum(1)
    
    # cosine distance
    alignment = sharks[i, 2:, :] * hunting 
    alignment /= torch.sqrt((sharks[i, 2:, :]**2).sum(0, keepdim=True))
    alignment = alignment.sum(0, keepdim=True)
                    
    # shark is exclusively moved by chasing fish through sprints
    shark_drive = hunting*(alignment>ALIGN_THRESH)*(avg_closest_dist<ATTACK_RADIUS) + hunting*REST_SPEED
    
    return shark_drive, dist_scores

#kill funtion
# function to check what fish have been killed in the current iteration
def check_kills(i, t, print_kills, kill_times, fs_inv_dist_norms):
    
    # let simulation settle before sharks are allowed to kill
    if i < HUNT_PERIOD:
        return kill_times

    # indexes of sharks who are outside of eating latency 
    ready_shark_idx = torch.where(kill_times[t, i:i+EATING_LATENCY, :].sum(0) == 0)[0]
    
    # indexes of fish who are still alive 
    live_fish_idx = torch.where(live_fish_mask[i])[0] 
    
    # filter fs_inv_dist_norms by these indexes
    filtered_fs_inv_dist_norms = torch.index_select(torch.index_select(fs_inv_dist_norms, 1, live_fish_idx), 2, ready_shark_idx)
    
    # get dist and idx of closest fish to each shark (means a shark can only get a single kill in an iteration)
    min_fish_dist, nearest_fish_idx = torch.min(1/filtered_fs_inv_dist_norms, dim=1) 
  
    # check if these fish are within death radius
    dead_fish = min_fish_dist < DEATH_RADIUS
    
    if dead_fish.any():
        # update live fish mask
        live_fish_mask[i+1, live_fish_idx[nearest_fish_idx[dead_fish]]] = False 
        # record kill for sharks responsible
        kill_times[t, i+EATING_LATENCY, ready_shark_idx[dead_fish[0]]] = True 
        
        if print_kills:
            print(f'Total kills: {kill_times.sum().item()}', end='\r')

    # also mask out the fish that were killed during the previous iterations
    live_fish_mask[i+1] *= live_fish_mask[i]
    
    return kill_times

#simulations
def run(trials=2):
    global fish, sharks, live_fish_mask, fish_colourmap
    fish_colourmap = torch.zeros((ITERS, N_FISH))
    kill_times = torch.zeros(trials, ITERS+EATING_LATENCY, N_SHARKS, dtype=torch.bool) # eating latency buffer removed at end
    attack_times = torch.zeros(trials, ITERS, N_SHARKS, dtype=torch.bool)
    for t in tqdm(range(trials), desc='Percentage of Trials Completed'):
        # initialise fish and shark
        # compute the vector components as the difference between the random numbers, then normalise them
        fish = torch.randn((ITERS, 4, N_FISH)) * SIZE/4
        live_fish_mask = torch.ones((ITERS, N_FISH))==True
        normalise(fish,0)
        sharks = torch.randn((ITERS, 4, N_SHARKS)) * SIZE/4
        normalise(sharks,0)
        
        # main update loop of the simulation
        for i in range(ITERS-1):
            
            # end trial if no remaining live fish
            n_dead_fish = live_fish_mask[i,:].size()[0]-live_fish_mask[i,:].sum()
            if N_FISH - n_dead_fish==0:
                break
            # update the velocity and position of shark and fish, then add some randomness
            fish[i+1, :2] = fish[i, :2] + MAX_FISH_VEL*fish[i, 2:]*live_fish_mask[i][None]
            fish[i+1, 2:] = fish[i, 2:]
            sharks[i+1, :2] = sharks[i, :2] + MAX_SHARK_VEL*sharks[i, 2:]
            # not adding noise to sharks because it interferes with resting speed, noise gets propagated from fish already
            sharks[i+1, 2:] = sharks[i, 2:]

            # ensure they do not overflow and in case bring them back to the other side
            fix_world_overflow(fish, i)
            fix_world_overflow(sharks, i)

            # measure the forces that act on fish and shark
            # get fish-fish and shark-fish distances
            fs_dist_vecs, fs_inv_dist_norms = get_distances(fish, sharks, i)
            fish_drive = get_fish_forces(i, fs_dist_vecs, fs_inv_dist_norms)
            shark_drive, dist_scores = get_shark_forces(i, fs_dist_vecs, fs_inv_dist_norms)

            # colourmap for fish
            fish_colourmap[i+1,:] = get_fish_colourmap(dist_scores)

            # check for killed fish, the second argument simply controls whether to print when a kill happens
            kill_times = check_kills(i, t, True, kill_times, fs_inv_dist_norms)

            # update the velocities of fish and shark through their momentum parameters
            # higher momentum means that they update slowly
            fish[i+1, 2:] = FISH_BETA*fish[i+1, 2:] + (1-FISH_BETA)*fish_drive
            sharks[i+1, 2:] = SHARK_BETA*sharks[i+1, 2:] + (1-SHARK_BETA)*shark_drive
            
            # if velocity is more than 20% greater than rest speed then define as attacking
            attack_times[t, i, :] = torch.linalg.norm(sharks[i+1, 2:], axis=0) > REST_SPEED*1.2
    
    # remove eating latency buffer from kill_times
    kill_times = kill_times[:, EATING_LATENCY:, :]
    
    print(f'\nAverage kills per trial: {str(kill_times.sum().item()/trials)}')
          
    return kill_times, attack_times





# Initialize positions and velocities
#fish_positions = np.random.rand(num_fish, 2)
#fish_velocities = np.random.rand(num_fish, 2) * 2 - 1
#shark_position = np.random.rand(1, 2)
#shark_velocity = np.random.rand(1, 2) * 2 - 1
 
# Normalize velocities
#fish_velocities /= np.linalg.norm(fish_velocities, axis=1)[:, np.newaxis]
#shark_velocity /= np.linalg.norm(shark_velocity, axis=1)
 
#def update(frame):
#    global fish_positions, fish_velocities, shark_position, shark_velocity
# 
#    # Update fish positions and velocities
#    for i in range(num_fish):
#        # Cohesion - move towards the center of mass( just to show where it should move to)
#       center_of_mass = np.mean(fish_positions, axis=0)
#        direction_to_center = center_of_mass - fish_positions[i]
#        fish_velocities[i] += cohesion_strength * direction_to_center
# 
#        # Avoidance of the shark
#        if np.linalg.norm(fish_positions[i] - shark_position) < avoidance_radius:
#            fish_velocities[i] = fish_positions[i] - shark_position
# 
#        # Nvel and position
#        fish_velocities[i] /= np.linalg.norm(fish_velocities[i])
#        fish_positions[i] += fish_velocities[i] * fish_speed
# 
#    # Shark behavior
#    shark_direction = center_of_mass - shark_position
#    shark_velocity = shark_direction / np.linalg.norm(shark_direction)
#    shark_position += shark_velocity * shark_speed
# 
#    # boundary conditions
#    fish_positions = np.mod(fish_positions, 1)
#    shark_position = np.mod(shark_position, 1)
# 
    # Plot
    plt.cla()
    plt.scatter(fish_positions[:, 0], fish_positions[:, 1], color='blue', s=10, label='Fish')
    plt.scatter(shark_position[:, 0], shark_position[:, 1], color='red', s=50, label='Shark')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("School of Tilapia and a Shark Predator")
    plt.legend()
    plt.axis('off')
    
    

 
# animation
kill_times, attack_times = run()

# animates the ocean simulation as a quiver plot
fig = plt.figure(figsize=(10,10))
plt.xlim(-SIZE/2, SIZE/2)
plt.ylim(-SIZE/2, SIZE/2)
i=0
fish_quiver = plt.quiver([1],[1])
shark_quiver = plt.quiver([1],[1])
scatter = plt.scatter([],[])
death_count = plt.gca().text(-50, 52,'')
fish = fish.cpu()
sharks = sharks.cpu()

def updatefig(*args):
    global i, fish_quiver, shark_quiver, scatter
    fish_quiver.remove()
    shark_quiver.remove()
    scatter.remove()
    live_fish, curr_sharks, dead_pos, curr_fish_colourmap = get_ocean(i)
    total_kills = N_FISH - live_fish.size(dim=1)   
    if N_SHARKS == 1:
        attack_status = 'Attacking' if attack_times[:, i, :] == True else 'Searching'
        death_count.set_text(f'Death Count: {total_kills}, Attack Status: {attack_status}, Iteration: {i}')
    else:
        death_count.set_text(f'Death Count: {total_kills}, Iteration: {i}')
    fish_quiver = plt.quiver(live_fish[0], live_fish[1], live_fish[2], live_fish[3], curr_fish_colourmap, cmap=custom_fish_cmap, scale=30, width=0.004) 
    shark_quiver = plt.quiver(curr_sharks[0], curr_sharks[1], curr_sharks[2], curr_sharks[3], color=np.where(attack_times[0,i,:].cpu()==True, 'red','orange'), scale=30, width=0.008)
    scatter = plt.scatter(dead_pos[0], dead_pos[1], marker='+', color='black')
    if (i<ITERS-1):
        i += 1
    else:
        i=0
    return fish_quiver, shark_quiver, scatter,

anim = FuncAnimation(fig,updatefig,frames=tqdm(range(ITERS-1), desc='Animation Completion'),interval=80,blit=True,repeat=True)
HTML(anim.to_jshtml())

#fig = plt.figure(figsize=(8, 8))
#ani = FuncAnimation(fig, update, frames=200, interval=100)
 
plt.show()
