# Test script to be able to find CAV corresponding to concept superset

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
from custom_ace import ConceptDiscovery
import argparse


def main(args):

    ###### related DIRs on CNS to store results #######
    discovered_concepts_dir = os.path.join(args.working_dir, 'concepts/')
    results_dir = os.path.join(args.working_dir, 'results/')
    cavs_dir = os.path.join(args.working_dir, 'cavs/')
    activations_dir = os.path.join(args.working_dir, 'acts/')
    results_summaries_dir = os.path.join(args.working_dir, 'results_summaries/')
    ###### Skip this to avoid removing existing concepts ######
    # if tf.gfile.Exists(args.working_dir):
        # tf.gfile.DeleteRecursively(args.working_dir)
    if not tf.gfile.Exists(args.working_dir):
        tf.gfile.MakeDirs(args.working_dir)
        tf.gfile.MakeDirs(discovered_concepts_dir)
        tf.gfile.MakeDirs(results_dir)
        tf.gfile.MakeDirs(cavs_dir)
        tf.gfile.MakeDirs(activations_dir)
        tf.gfile.MakeDirs(results_summaries_dir)
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
    # cd.create_patches(param_dict={'n_segments': [15, 50, 80]}) <- SKIP THIs
    # Saving the concept discovery target class images
    # image_dir = os.path.join(discovered_concepts_dir, 'images', args.target_class) <- SKIP THIs
    # tf.gfile.MakeDirs(image_dir)  <- SKIP THIs
    # ace_helpers.save_images(image_dir,
                            # (cd.discovery_images * 256).astype(np.uint8))  <- SKIP THIs
    # Discovering Concepts
    # cd.discover_concepts(method='KM', param_dicts={'n_clusters': 25})  <- SKIP THIs
    # del cd.dataset  # Free memory <- SKIP THIs
    # del cd.image_numbers <- SKIP THIs
    # del cd.patches <- SKIP THIs
    # Save discovered concept images (resized and original sized)
    # ace_helpers.save_concepts(cd, discovered_concepts_dir)
    # Calculating CAVs and TCAV scores

    # Load concept super-pixels
    concept_images_dirs = [
        'mixed8_ambulance_concept7',
        'mixed8_jeep_concept3',
        'mixed8_jeep_concept17'
    ]

    # images = [glob(f'{os.path.join(discovered_concepts_dir, concept_images_dir)}/*.png') for concept_images_dir in concept_images_dirs]
    images = [glob(f'{os.path.join(discovered_concepts_dir, concept_images_dir)}/') for concept_images_dir in concept_images_dirs]
    images = [image for nested_images in images for image in nested_images]

    cd.dic = {} # not initialised elsewhere as skip concept discovery steps
    for bn in cd.bottlenecks:
        cd.dic[bn] = {}
        cd.dic[bn]['concepts'] = ['combined_concept']
        cd.dic[bn]['combined_concept'] =  {}
        cd.dic[bn]['combined_concept']['images'] = images

    cav_accuracies = cd.cavs(min_acc=0.0) # <- skip most of cavs method - only want concept 
    print(cav_accuracies)

    # TODO: Need to update how TCAV scores are created
    # scores = cd.tcavs(test=False)
    # Save ACE report <- Skip for now
    # ace_helpers.save_ace_report(cd, cav_accuraciess, scores,
    #                                 results_summaries_dir + f'{args.bottlenecks}_{args.target_class}_ace_results.txt')
    # # Plot examples of discovered concepts <- Skip for now
    # for bn in cd.bottlenecks:
    #     ace_helpers.plot_concepts(cd, bn, args.target_class, 10, address=results_dir)
    # # Delete concepts that don't pass statistical testing
    # cd.test_and_remove_concepts(scores)

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
  parser.add_argument('--model_path', type=str,
      help='Path to model checkpoints.', default='./tensorflow_inception_graph.pb')
  parser.add_argument('--labels_path', type=str,
      help='Path to model checkpoints.', default='./imagenet_labels.txt')
  parser.add_argument('--target_class', type=str,
      help='The name of the target class to be interpreted', default='dumbbell')
  parser.add_argument('--bottlenecks', type=str,
      help='Names of the target layers of the network (comma separated)',
                      default='mixed5b')
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

    # Compare ambulance concept7 and jeep concept3 maybe concept17
    # Corresponds to wheels and tyres
    # Ambulance, concept7 TCAV: 0.59875
    # Jeep, concept3 TCAV: 0.33
    # Jeep, concept17 TCAV:0.46 (does not visually correspond to ambulance concept as closely but has higher TCAV score)

    img_code = 'n03255030' # <- shouldn't be used
    # Copy random discovery folders to image directory
    for random_dir in glob('./ImageNet/random*'):
        random_dir_name = random_dir.split('/')[-1]
        copy_tree(random_dir, f'./ImageNet/ILSVRC2012_img_train/{img_code}/img_sample/{random_dir_name}')

    args = parse_arguments(sys.argv[1:])
    args.model_to_run = 'InceptionV3'
    args.model_path = './inception_v3.h5'
    args.bottlenecks = 'mixed8'
    args.source_dir = 'dummy' # <- shouldn't be used
    args.target_class = 'dummy' # <- shouldn't be used

    main(args)

    # Delete random images
    for dir in glob(f'./ImageNet/ILSVRC2012_img_train/{img_code}/img_sample/random*'):
        rmtree(dir)


