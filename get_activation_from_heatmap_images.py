"""This script runs the whole ACE method."""


import sys
import os
import numpy as np
import sklearn.metrics as metrics
from tcav import utils
import tensorflow as tf

import ace_helpers
from ace import ConceptDiscovery
import argparse


def main(args):

  ###### related DIRs on CNS to store results #######
  cavs_dir = os.path.join(args.working_dir, 'cavs/')
  activations_dir = os.path.join(args.working_dir, 'acts/')
  if tf.gfile.Exists(args.working_dir):
    tf.gfile.DeleteRecursively(args.working_dir)
  tf.gfile.MakeDirs(args.working_dir)
  tf.gfile.MakeDirs(cavs_dir)
  tf.gfile.MakeDirs(activations_dir)
  random_concept = 'random_discovery'  # Random concept for statistical testing
  sess = utils.create_session()
  mymodel = ace_helpers.make_model(
      sess, args.model_to_run, args.model_path, args.labels_path)
  # Creating the ConceptDiscovery class instance
  cd = ConceptDiscovery(
      mymodel,
      args.target_class,
      random_concept,
      args.bottlenecks.split(','),
      sess,
      args.source_dir,
      activations_dir,
      cavs_dir,
      num_random_exp=args.num_random_exp,
      channel_mean=True,
      max_imgs=args.max_imgs,
      min_imgs=args.min_imgs,
      num_discovery_imgs=args.max_imgs,
      num_workers=args.num_parallel_workers)


  # For given set of images find the bottleneck activation
  # For bottleneck activations, find pairwise cosine similarity

  # Discovering Concepts
  # TODO: Still need to crop and resize images found through occlusion to get superpixels
  activations, img_filenames = cd.get_activations()

  # For the 17 images found, expect 17C2 = 136 pairs
  cosine_sims = []
  for i in range(len(activations)-1):
      for j in range(i+1,len(activations)):
          cosine_sim = {}
          cosine_sim['image_1'] = i
          cosine_sim['image_2'] = j
          cosine_sim['cosine_sim'] = ace_helpers.cosine_similarity(activations[i],activations[j])
          cosine_sims.append(cosine_sim)

  print(sorted(cosine_sims, key=lambda k: k['cosine_sim'], reverse=True)[:3])
  print(img_filenames[sorted(cosine_sims, key=lambda k: k['cosine_sim'], reverse=True)[0]['image_1']])
  print(img_filenames[sorted(cosine_sims, key=lambda k: k['cosine_sim'], reverse=True)[0]['image_2']])
  print('-------------------------')
  print(sorted(cosine_sims, key=lambda k: k['cosine_sim'])[:3])
  print(img_filenames[sorted(cosine_sims, key=lambda k: k['cosine_sim'])[0]['image_1']])
  print(img_filenames[sorted(cosine_sims, key=lambda k: k['cosine_sim'])[0]['image_2']])

  # Compare and contrast with randomly selected images 


#   del cd.dataset  # Free memory
#   del cd.image_numbers
#   del cd.patches

  # TODO: Save discovered concepts   
  # Save discovered concept images (resized and original sized)
#   ace_helpers.save_concepts(cd, discovered_concepts_dir)

def parse_arguments(argv):
  """Parses the arguments passed to the run.py script."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_dir', type=str,
      help='''Directory where the network's classes image folders and random
      concept folders are saved.''', default='../inm363-individual-project/test_images')
  parser.add_argument('--working_dir', type=str,
      help='Directory to save the results.', default='./ACE')
  parser.add_argument('--model_to_run', type=str,
      help='The name of the model.', default='GoogleNet')
  parser.add_argument('--model_path', type=str,
      help='Path to model checkpoints.', default='./tensorflow_inception_graph.pb')
  parser.add_argument('--labels_path', type=str,
      help='Path to model checkpoints.', default='./imagenet_labels.txt')
  parser.add_argument('--target_class', type=str,
      help='The name of the target class to be interpreted', default='bubble')
  parser.add_argument('--bottlenecks', type=str,
      help='Names of the target layers of the network (comma separated)',
                      default='mixed4c')
  parser.add_argument('--num_random_exp', type=int,
      help="Number of random experiments used for statistical testing, etc",
                      default=20)
  parser.add_argument('--max_imgs', type=int,
      help="Maximum number of images in a discovered concept",
                      default=40)
  parser.add_argument('--min_imgs', type=int,
      help="Minimum number of images in a discovered concept",
                      default=40)
  parser.add_argument('--num_parallel_workers', type=int,
      help="Number of parallel jobs.",
                      default=0)
  return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

