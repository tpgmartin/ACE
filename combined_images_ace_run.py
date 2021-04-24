"""This script runs the whole ACE method."""

from distutils.dir_util import copy_tree
from glob import glob
import numpy as np
import os
import pandas as pd
import sklearn.metrics as metrics
from shutil import copyfile, rmtree
import sys
from tcav import utils
import tensorflow as tf

import ace_helpers
from combined_images_ace import ConceptDiscovery
import argparse


def main(args):

  ###### related DIRs on CNS to store results #######
  discovered_concepts_dir = os.path.join(args.working_dir, 'concepts/')
  results_dir = os.path.join(args.working_dir, 'results/')
  cavs_dir = os.path.join(args.working_dir, 'cavs/')
  activations_dir = os.path.join(args.working_dir, 'acts/')
  results_summaries_dir = os.path.join(args.working_dir, 'results_summaries/')
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
  # Creating the dataset of image patches
  cd.create_patches(param_dict={'n_segments': [15, 50, 80]})

  # Saving the concept discovery target class images
  image_dir = os.path.join(discovered_concepts_dir, 'images', args.target_class)
  tf.gfile.MakeDirs(image_dir)
  ace_helpers.save_images(image_dir,
                            (cd.discovery_images * 256).astype(np.uint8))
  ace_helpers.save_images_lookup(cd.discovery_image_filenames, args.target_class, args.label_mapping)
  # Discovering Concepts
  cd.discover_concepts(method='KM', param_dicts={'n_clusters': 25})
  del cd.dataset  # Free memory
  del cd.image_numbers
  del cd.patches
  # Save discovered concept images (resized and original sized)
  ace_helpers.save_concepts(cd, discovered_concepts_dir)
  # Calculating CAVs and TCAV scores
  cav_accuraciess = cd.cavs(min_acc=0.0)
  scores = cd.tcavs(test=False)
  ace_helpers.save_ace_report(cd, cav_accuraciess, scores,
                                    results_summaries_dir + f'{args.bottlenecks}_{args.target_class}_ace_results.txt')
  # Plot examples of discovered concepts
  for bn in cd.bottlenecks:
    ace_helpers.plot_concepts(cd, bn, args.target_class, 10, address=results_dir)
  # Delete concepts that don't pass statistical testing
  cd.test_and_remove_concepts(scores)

def parse_arguments(argv):
  """Parses the arguments passed to the run.py script."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_dir', type=str,
      help='''Directory where the network's classes image folders and random
      concept folders are saved.''', default='./ImageNet')
  parser.add_argument('--working_dir', type=str,
      help='Directory to save the results.', default='./ACE')
  parser.add_argument('--model_to_run', type=str,
      help='The name of the model.', default='GoogleNet')
      # help='The name of the model.', default='InceptionV3')
  parser.add_argument('--model_path', type=str,
      help='Path to model checkpoints.', default='./tensorflow_inception_graph.pb')
      # help='Path to model checkpoints.', default=None)
  parser.add_argument('--labels_path', type=str,
      help='Path to model checkpoints.', default='./imagenet_labels.txt')
  parser.add_argument('--target_class', type=str,
      help='The name of the target class to be interpreted', default='dumbbell')
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

    args = parse_arguments(sys.argv[1:])
    combined_label = 'ambulance_and_police_van'

    samples = [
        '../inm363-individual-project/baseline_prediction_samples/ambulancebaseline_prediction_samples.csv',
        # '../inm363-individual-project/baseline_prediction_samples/jeepbaseline_prediction_samples.csv',
        # '../inm363-individual-project/baseline_prediction_samples/bullet_trainbaseline_prediction_samples.csv'
        '../inm363-individual-project/baseline_prediction_samples/police_vanbaseline_prediction_samples.csv'
    ]

    df = pd.concat([pd.read_csv(sample).iloc[:(args.max_imgs//2),:] for sample in samples])
    true_label = combined_label
    sample_dir_path = f'./ImageNet/ILSVRC2012_img_train/{combined_label}/img_sample'
    filepaths = df['filename']

    # Copy sample images 
    img_sample_dir = f'./ImageNet/ILSVRC2012_img_train/{combined_label}/img_sample/{combined_label}'
    os.makedirs(img_sample_dir, exist_ok=True)
    for f in filepaths:
        filename = f.split('/')[-1]
        original_filepath = './' + '/'.join(f.split('/')[2:])
        copyfile(original_filepath, f'{img_sample_dir}/{filename}')

    # Copy random discovery folders to image directory
    for random_dir in glob('./ImageNet/random*'):
        random_dir_name = random_dir.split('/')[-1]
        copy_tree(random_dir, f'./ImageNet/ILSVRC2012_img_train/{combined_label}/img_sample/{random_dir_name}')

    
    args.model_to_run = 'InceptionV3'
    args.model_path = './inception_v3.h5'
    args.bottlenecks = 'mixed8'

    args.source_dir = sample_dir_path
    args.target_class = combined_label
    args.label_mapping = {
        'n02701002': 'ambulance',
        'n03977966': 'police_van'
    }

    main(args)

    # Delete random images
    for dir in glob(f'./ImageNet/ILSVRC2012_img_train/{combined_label}/img_sample/random*'):
        rmtree(dir)
