import sys
import os
cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../deepg/code/')
import numpy as np
from eran import ERAN
from read_net_file import *
from read_zonotope_file import read_zonotope
import tensorflow as tf
import csv
import random
import cv2
import time
from tqdm import tqdm
from ai_milp import *
import argparse
from config import config
from constraint_utils import *
import re
import itertools
from multiprocessing import Pool, Value
import onnxruntime.backend as rt
import logging
#ZONOTOPE_EXTENSION = '.zt'
EPS = 10**(-9)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def isnetworkfile(fname):
    _, ext = os.path.splitext(fname)
    if ext not in ['.pyt', '.meta', '.tf','.onnx', '.pb']:
        raise argparse.ArgumentTypeError('only .pyt, .tf, .onnx, .pb, and .meta formats supported')
    return fname



def parse_input_box(text):
    intervals_list = []
    for line in text.split('\n'):
        if line!="":
            interval_strings = re.findall("\[-?\d*\.?\d+, *-?\d*\.?\d+\]", line)
            intervals = []
            for interval in interval_strings:
                interval = interval.replace('[', '')
                interval = interval.replace(']', '')
                [lb,ub] = interval.split(",")
                intervals.append((np.double(lb), np.double(ub)))
            intervals_list.append(intervals)

    # return every combination
    boxes = itertools.product(*intervals_list)
    return boxes


def show_ascii_spec(lb, ub, n_rows, n_cols, n_channels):
    print('==================================================================')
    for i in range(n_rows):
        print('  ', end='')
        for j in range(n_cols):
            print('#' if lb[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
        print('  |  ', end='')
        for j in range(n_cols):
            print('#' if ub[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
        print('  |  ')
    print('==================================================================')


def normalize_poly(num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights, uexpr_dim, means, stds, dataset):
    # normalization taken out of the network
    if dataset == 'mnist' or dataset == 'fashion':
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[0]) / stds[0]
            uexpr_cst[i] = (uexpr_cst[i] - means[0]) / stds[0]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[0]
            uexpr_weights[i] /= stds[0]
    else:
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[i % 3]) / stds[i % 3]
            uexpr_cst[i] = (uexpr_cst[i] - means[i % 3]) / stds[i % 3]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[(i // num_params) % 3]
            uexpr_weights[i] /= stds[(i // num_params) % 3]


def denormalize(image, means, stds, dataset):
    if dataset == 'mnist'  or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = image[i]*stds[0] + means[0]
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = image[count]*stds[0] + means[0]
            count = count + 1
            tmp[count] = image[count]*stds[1] + means[1]
            count = count + 1
            tmp[count] = image[count]*stds[2] + means[2]
            count = count + 1

        for i in range(3072):
            image[i] = tmp[i]

def upsamplefeatures(input):
    N, Cin, Hin, Win = input.shape
    sca = 2
    sca2 = sca*sca
    Cout = Cin//sca2
    Hout = Hin*sca
    Wout = Win*sca
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert (Cin%sca2 == 0), 'Invalid input dimensions: number of channels should be divisible by 4'
    result = np.zeros((N, Cout, Hout, Wout), dtype=np.float32)
    for idx in range(sca2):
        result[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca] = input[:, idx:Cin:sca2, :, :]
    return result

def get_tests(dataset, geometric):
    if geometric:
        csvfile = open('../deepg/code/datasets/{}_test.csv'.format(dataset), 'r')
    else:
        csvfile = open('../data/{}_test.csv'.format(dataset), 'r')
    tests = csv.reader(csvfile, delimiter=',')
    return tests


def init_domain(d):
    if d == 'refinezono':
        return 'deepzono'
    elif d == 'refinepoly':
        return 'deeppoly'
    else:
        return d

def variable_to_cv2_image(varim):
    res = (varim * 255.).clip(0, 255).astype(np.uint8)	
    return res

def normalize_ffd(data):
    """Normalizes a unit8 image to a float32 image in the range [0, 1]

	Args:
		data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
		The normalization function from ffdnet
	"""
    return np.float32(data/255.)

def concatenate_input_noise_map(input, noise_sigma):
    N, C, H, W = input.shape
    sca = 2
    sca2 = sca*sca
    Cout = sca2*C
    Hout = H//sca
    Wout = W//sca
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]
    downsampledfeatures = np.zeros((N, Cout, Hout, Wout), dtype=np.float32)
    # The N above equals to 1
    noise_map = np.full((1, C, Hout, Wout), noise_sigma, dtype=np.float32)
    print(downsampledfeatures.shape)
    print(noise_map.shape)
    for idx in range(sca2):
        downsampledfeatures[:, idx:Cout:sca2, :, :] = input[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca]
    return np.concatenate((noise_map, downsampledfeatures), axis=1)

parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--netname', type=isnetworkfile, default=config.netname, help='the network name, the extension can be only .pb, .pyt, .tf, .meta, and .onnx')
parser.add_argument('--epsilon', type=float, default=config.epsilon, help='the epsilon for L_infinity perturbation')
############## My parser
parser.add_argument('--small_epsilon', type=float, default=None, help='the epsilon for small perturbation pixels')
parser.add_argument('--sig_beta', type=float, default=None, help='the beta for significant perturbation pixels')
parser.add_argument('--left_bottom_corner', type=int, nargs=2, default=None, help='the left botton corner of the rectangular region, x axis and y axis')
parser.add_argument('--right_up_corner', type=int, nargs=2, default= None, help='the right up corner of the rectangular region, x axis and y axis')
parser.add_argument('--search_region', type=str2bool, default=None,  help='flag specifying where to search the significant perturbed region or not')
parser.add_argument('--fb_threshold', type=float, default=None, help='the threshold to distinguish the region')
parser.add_argument('--region_above_thre', type=str2bool, default=None, help='the region is the part above the threshold or below the threshold')
parser.add_argument('--savepath', type=str, default=None, help='csv filename to store the experiment result')
parser.add_argument('--ffdnet', type=str2bool, default=False,  help='flag to set deeppoly to over-approximate ffdnet')
parser.add_argument('--is_rgb', type=str2bool, default=False,  help='flag to that the image is rgb or not')
parser.add_argument("--noise_sigma", type=float, default=25, help='Denoise level used on FFDNet')
parser.add_argument('--two_lbs', type=str2bool, default=False,  help='flag to allow two symbolic lower bounds or not')
parser.add_argument('--random_prune', type=str2bool, default=False,  help='flag to allow random expression prune or not')
############ End of my parserssss
parser.add_argument('--zonotope', type=str, default=config.zonotope, help='file to specify the zonotope matrix')
parser.add_argument('--specnumber', type=int, default=config.specnumber, help='the property number for the acasxu networks')
parser.add_argument('--domain', type=str, default=config.domain, help='the domain name can be either deepzono, refinezono, deeppoly or refinepoly')
parser.add_argument('--dataset', type=str, default=config.dataset, help='the dataset, can be either mnist, cifar10, acasxu, or fashion')
parser.add_argument('--complete', type=str2bool, default=config.complete,  help='flag specifying where to use complete verification or not')
parser.add_argument('--timeout_lp', type=float, default=config.timeout_lp,  help='timeout for the LP solver')
parser.add_argument('--timeout_milp', type=float, default=config.timeout_milp,  help='timeout for the MILP solver')
parser.add_argument('--numproc', type=int, default=config.numproc,  help='number of processes for MILP / LP / k-ReLU')
parser.add_argument('--use_default_heuristic', type=str2bool, default=config.use_default_heuristic,  help='whether to use the area heuristic for the DeepPoly ReLU approximation or to always create new noise symbols per relu for the DeepZono ReLU approximation')
parser.add_argument('--use_milp', type=str2bool, default=config.use_milp,  help='whether to use milp or not')
parser.add_argument('--dyn_krelu', action='store_true', default=None, help='dynamically select parameter k')
parser.add_argument('--use_2relu', action='store_true', default=None, help='use 2-relu')
parser.add_argument('--use_3relu', action='store_true', default=None, help='use 3-relu')
parser.add_argument('--mean', nargs='+', type=float, default=config.mean, help='the mean used to normalize the data with')
parser.add_argument('--std', nargs='+', type=float, default=config.std, help='the standard deviation used to normalize the data with')
parser.add_argument('--data_dir', type=str, default=config.data_dir, help='data location')
parser.add_argument('--geometric_config', type=str, default=config.geometric_config, help='config location')
parser.add_argument('--num_params', type=int, default=config.num_params, help='Number of transformation parameters')
parser.add_argument('--num_tests', type=int, default=config.num_tests, help='Number of images to test')
parser.add_argument('--from_test', type=int, default=config.from_test, help='Number of images to test')
parser.add_argument('--debug', action='store_true', default=config.debug, help='Whether to display debug info')
parser.add_argument('--attack', action='store_true', default=config.attack, help='Whether to attack')
parser.add_argument('--geometric', '-g', dest='geometric', default=config.geometric, action='store_true', help='Whether to do geometric analysis')
parser.add_argument('--input_box', default=config.input_box,  help='input box to use')
parser.add_argument('--output_constraints', default=config.output_constraints, help='custom output constraints to check')


# Logging options
parser.add_argument('--logdir', type=str, default=None, help='Location to save logs to. If not specified, logs are not saved and emitted to stdout')
parser.add_argument('--logname', type=str, default=None, help='Directory of log files in `logdir`, if not specified timestamp is used')


args = parser.parse_args() 
# The return value from parse_args() is a Namespace containing the arguments to the command. The object holds the argument values as attributes
for k, v in vars(args).items():
    setattr(config, k, v) #takes three parameters:object whose attributes to be set, attribute name, attribute name
config.json = vars(args)

if config.specnumber and not config.input_box and not config.output_constraints:
    config.input_box = '../data/acasxu/specs/acasxu_prop_' + str(config.specnumber) + '_input_prenormalized.txt'
    config.output_constraints = '../data/acasxu/specs/acasxu_prop_' + str(config.specnumber) + '_constraints.txt'

assert config.netname, 'a network has to be provided for analysis.'

#if len(sys.argv) < 4 or len(sys.argv) > 5:
#    print('usage: python3.6 netname epsilon domain dataset')
#    exit(1)

netname = config.netname
filename, file_extension = os.path.splitext(netname)

is_trained_with_pytorch = file_extension==".pyt"
is_saved_tf_model = file_extension==".meta"
is_pb_file = file_extension==".pb"
is_tensorflow = file_extension== ".tf"
is_onnx = file_extension == ".onnx"
assert is_trained_with_pytorch or is_saved_tf_model or is_pb_file or is_tensorflow or is_onnx, "file extension not supported"

epsilon = config.epsilon
assert (epsilon >= 0) and (epsilon <= 1), "epsilon can only be between 0 and 1"

zonotope_file = config.zonotope
zonotope = None
zonotope_bool = (zonotope_file!=None)
if zonotope_bool:
    zonotope = read_zonotope(zonotope_file)

domain = config.domain

if zonotope_bool:
    assert domain in ['deepzono', 'refinezono'], "domain name can be either deepzono or refinezono"
elif not config.geometric:
    assert domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly'], "domain name can be either deepzono, refinezono, deeppoly or refinepoly"

dataset = config.dataset

if zonotope_bool==False:
   assert dataset in ['mnist', 'cifar10', 'acasxu', 'fashion'], "only mnist, cifar10, acasxu, and fashion datasets are supported"

constraints = None
if config.output_constraints:
    constraints = get_constraints_from_file(config.output_constraints)

mean = 0
std = 0
is_conv = False

complete = (config.complete==True)

if(dataset=='acasxu'):
    print("netname ", netname, " specnumber ", config.specnumber, " domain ", domain, " dataset ", dataset, "args complete ", config.complete, " complete ",complete, " timeout_lp ",config.timeout_lp)
else:
    print("netname ", netname, " epsilon ", epsilon, " domain ", domain, " dataset ", dataset, "args complete ", config.complete, " complete ",complete, " timeout_lp ",config.timeout_lp)

non_layer_operation_types = ['NoOp', 'Assign', 'Const', 'RestoreV2', 'SaveV2', 'PlaceholderWithDefault', 'IsVariableInitialized', 'Placeholder', 'Identity']

if is_saved_tf_model or is_pb_file:
    netfolder = os.path.dirname(netname)

    tf.logging.set_verbosity(tf.logging.ERROR)

    sess = tf.Session()
    if is_saved_tf_model:
        saver = tf.train.import_meta_graph(netname)
        saver.restore(sess, tf.train.latest_checkpoint(netfolder+'/'))
    else:
        with tf.gfile.GFile(netname, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.graph_util.import_graph_def(graph_def, name='')
    ops = sess.graph.get_operations()
    last_layer_index = -1
    while ops[last_layer_index].type in non_layer_operation_types:
        last_layer_index -= 1
    eran = ERAN(sess.graph.get_tensor_by_name(ops[last_layer_index].name + ':0'), sess)

else:
    if(zonotope_bool==True):
        num_pixels = len(zonotope)
    elif(dataset=='mnist'):
        num_pixels = 784
    elif (dataset=='cifar10'):
        num_pixels = 3072
    elif(dataset=='acasxu'):
        num_pixels = 5
    if is_onnx:
        model, is_conv = read_onnx_net(netname)
        # this is to have different defaults for mnist and cifar10
    else:
        model, is_conv, means, stds = read_tensorflow_net(netname, num_pixels, is_trained_with_pytorch)
    eran = ERAN(model, is_onnx=is_onnx)

if not is_trained_with_pytorch:
    if dataset == 'mnist' and not config.geometric:
        means = [0]
        stds = [1]
    elif dataset == 'acasxu':
        means = [1.9791091e+04,0.0,0.0,650.0,600.0]
        stds = [60261.0,6.28318530718,6.28318530718,1100.0,1200.0]
    else:
        means = [0.5, 0.5, 0.5]
        stds = [1, 1, 1]

is_trained_with_pytorch = is_trained_with_pytorch or is_onnx

if config.mean is not None:
    means = config.mean
    stds = config.std

correctly_classified_images = 0
verified_images = 0


if dataset:
    if config.input_box is None:
        tests = get_tests(dataset, config.geometric)
    else:
        tests = open(config.input_box, 'r').read()

# Merge ffdnet with deeppoly
#test_input = np.asarray([1,1], dtype=np.float64)
specLB = np.asarray([-1,-1,-1,-1], dtype=np.float64)
specUB = np.asarray([1,1,1,1], dtype=np.float64)
#specLB = np.copy(test_input)
#specUB = np.copy(test_input)
one_dim_LB = specLB.flatten()
one_dim_UB = specUB.flatten()
print("enter eran function")
start = time.time()
eran_result = eran.analyze_box(one_dim_LB, one_dim_UB, init_domain(domain), config.timeout_lp, config.timeout_milp, config.use_default_heuristic, args.two_lbs, args.random_prune)
print("leave eran function")
end = time.time()
nub = np.array(eran_result[3])
nlb = np.array(eran_result[2])
print(nlb[-1])
print(nub[-1])
print(end - start, "seconds")
fullpath="exeu_time.csv"
with open(fullpath, 'a+', newline='') as write_obj:
    csv_writer = csv.writer(write_obj)
    #csv_writer.writerow(list(nlb[-1]))
    #csv_writer.writerow(list(nub[-1]))
    net_name_list = netname.split("/")
    row = [net_name_list[-1], "Two_lbs:"+str(args.two_lbs),"ran_prune:"+str(args.random_prune), str(end - start)+" second"]
    csv_writer.writerow(row)
fullpath="new_method_bounds.csv"
with open(fullpath, 'a+', newline='') as write_obj:
    csv_writer = csv.writer(write_obj)
    csv_writer.writerow([net_name_list[-1], "Two_lbs:"+str(args.two_lbs),"ran_prune:"+str(args.random_prune),"Lower"]+list(nlb[-1]))
    csv_writer.writerow([net_name_list[-1], "Two_lbs:"+str(args.two_lbs),"ran_prune:"+str(args.random_prune),"Upper"]+list(nub[-1]))
