import numpy as np
import os
import random
random.seed(1337)

def notempty(l):
    return not all([d==[] for d in l])
def makename(n):
    #return '/home/spiro/dtd/images/'+n.split('_')[0] +'/' + n
    return '/home/spiro/AlexNet/npz/relu5__home_spiro_dtd_images_' + n.split('_')[0] + '_' +  n
root = "../dtd/images/"
all_paths = [random.sample(os.listdir(root+d),len(os.listdir(root+d))) for d in os.listdir(root)]
paths = []
for i in range(4000//50):
    if all([d==[] for d in all_paths]):
        break
    all_paths = [p for p in all_paths if p!=[]]
    pairs = []
    pair_type = random.choice(['same','diff'])
    left = []
    right = []
    for j in range(50):
        try:
            category = random.choice(range(len(all_paths)))
            while all_paths[category]==[] and notempty(all_paths):
                category = random.choice(range(len(all_paths)))
            left_sample = all_paths[category].pop()
            if pair_type == 'same':
                right_sample = all_paths[category].pop()
            elif pair_type == 'diff':
                category = random.choice(range(len(all_paths)))
                while all_paths[category] == [] and notempty(all_paths):
                    category = random.choice(range(len(all_paths)))
                right_sample = all_paths[category].pop()
            left.append(makename(left_sample))
            #print makename(left_sample)
            right.append(makename(right_sample))
        except:
            pass
    paths.extend(left)
    paths.extend(right)
for p in paths:
    print p


#        if pair_type=='same':
#            left_cat = 
#        if [d==[] for d in all_paths].all():
#            break
#        left_sample = random.choice(all_paths[category])
#        all_paths[category].remove(left_sample)
#        right_sample = random.choice(all_paths[category])
#        all_paths[category].remove(right_sample)
#        left.append(left_sample)
#        right.append(right_sample)
#    
#
#def pick_two_categories(p_type):
#    category = random.choice(range(len(all_paths)))
#    while not all_paths[category]:
#        category = random.choice(range(len(all_paths)))
#    if pair_type == 'same':
#        left_sample = category,category
#    elif pair_type == 'diff':
#        second_category = random.choice(range(len(all_paths)))
#        while not all_paths[category]:
#            second_category = random.choice(range(len(all_paths)))
#        return category,second_category
