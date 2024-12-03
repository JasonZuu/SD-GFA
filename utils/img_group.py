from itertools import combinations
from random import choice


def build_triplet_combination(data_df):
     triplets = []
     more_face_data_df = data_df[data_df['person_img_num']>1]
     more_face_person_name = list(set(more_face_data_df['person_name'].values))
     for person_name in more_face_person_name:
          img_paths = more_face_data_df[more_face_data_df['person_name']==person_name]['img_path'].values
          other_group_img_paths = data_df[data_df['person_name']!=person_name]['img_path'].values
          img_tuples = combinations(img_paths, r=2)
          for img_tuple in img_tuples:
               other_group_img = choice(other_group_img_paths)
               img_triplet = list(img_tuple)
               img_triplet.append(other_group_img)
               triplets.append(img_triplet)
     return triplets


def build_triplet(data_df):
     triplets = []
     more_face_data_df = data_df[data_df['person_img_num']>1]
     more_face_person_name = list(set(more_face_data_df['person_name'].values))
     for person_name in more_face_person_name:
          img_paths = more_face_data_df[more_face_data_df['person_name']==person_name]['img_path'].values
          other_group_img_paths = data_df[data_df['person_name']!=person_name]['img_path'].values
          num_tuples = len(img_paths)-1
          for i_tuple in range(num_tuples):
               other_group_img = choice(other_group_img_paths)
               img_triplet = [img_paths[i_tuple], img_paths[i_tuple+1]]
               img_triplet.append(other_group_img)
               triplets.append(img_triplet)
     return triplets


def get_tuple_of_same_person_comb(data_df):
     tuples = []
     more_face_data_df = data_df[data_df['person_img_num']>1]
     more_face_person_name = list(set(more_face_data_df['person_name'].values))
     for person_name in more_face_person_name:
          img_paths = more_face_data_df[more_face_data_df['person_name']==person_name]['img_path'].values
          img_tuples = combinations(img_paths, r=2)
          tuples.extend(list(img_tuples))
     return tuples


def get_tuple_of_same_person(data_df):
     tuples = []
     more_face_data_df = data_df[data_df['person_img_num']>1]
     more_face_person_name = list(set(more_face_data_df['person_name'].values))
     for person_name in more_face_person_name:
          img_paths = more_face_data_df[more_face_data_df['person_name']==person_name]['img_path'].values
          num_tuples = len(img_paths)-1
          for i_tuple in range(num_tuples):
               img_tuple = [img_paths[i_tuple], img_paths[i_tuple+1]]
               tuples.append(img_tuple)
     return tuples