#!/usr/bin/env python

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from collections import defaultdict
import pprint

if len(sys.argv) < 2:
    print "USAGE: compare_perch_cnn.py <perch_results.txt> <cnn_results.txt>"
    exit(1)

PERCH_FILENAME = sys.argv[1]
CNN_FILENAME = sys.argv[2]
NUM_OBJECTS = 38

def PlotAccuracy(title, file_name, bin_min_items, bin_max_items):
    pp = pprint.PrettyPrinter(indent=4)
    mpl.rcParams['xtick.labelsize'] = 4

    f_cnn = open(CNN_FILENAME)
    f_perch = open(PERCH_FILENAME)

    csv_cnn = csv.reader(f_cnn)
    next(csv_cnn, None)
    csv_perch = csv.reader(f_perch)
    next(csv_perch, None)

    cnn_exp_map = defaultdict(dict)
    cnn_obj_map = defaultdict(list)

    # CSV FIELDS
    # PERCH: input,target_item,target_number,num_items,IOU,accuracy,run_time
    # CNN:   input,target_item,target_number,num_items,IOU,accuracy,score,run_time

    for row in csv_cnn:
        cnn_exp_map[row[0]][row[2]] = row[5]
        cnn_obj_map[row[2]].append(row[5])

    perch_exp_map = defaultdict(dict)
    perch_obj_map = defaultdict(list)

    for row in csv_perch:
        if float(row[3]) < bin_min_items or float(row[3]) > bin_max_items:
            continue
        perch_exp_map[row[0]][row[2]] = row[5]
        # perch_obj_map[row[2]].append(row[5])

    for exp, objs in perch_exp_map.iteritems():
        for obj, acc in objs.iteritems():
            perch_obj_map[obj].append([acc, cnn_exp_map[exp][obj]])

    avg_perch_accuracies = {}
    avg_cnn_accuracies = {}
    perch_accs = []
    cnn_accs = []
    perch_stddevs = []
    cnn_stddevs = []
    for obj in xrange(1,NUM_OBJECTS+1):
        obj = str(obj)
        if obj in perch_obj_map:
            perch_obj_accs =  [float(accs[0]) for accs in perch_obj_map[obj]]
            cnn_obj_accs = [float(accs[1]) for accs in perch_obj_map[obj]]
            avg_perch_accuracies[obj] = np.mean(perch_obj_accs)
            avg_cnn_accuracies[obj] = np.mean(cnn_obj_accs)
            perch_stddevs.append(np.std(perch_obj_accs, ddof=0))
            cnn_stddevs.append(np.std(cnn_obj_accs, ddof=0))

            perch_accs.append(avg_perch_accuracies[obj])
            cnn_accs.append(avg_cnn_accuracies[obj])
        else:
            avg_perch_accuracies[obj] = 0
            avg_cnn_accuracies[obj] = 0
            perch_accs.append(0)
            cnn_accs.append(0)
            perch_stddevs.append(0)
            cnn_stddevs.append(0)

    # pp.pprint(avg_perch_accuracies)
    # pp.pprint(avg_cnn_accuracies)

    width = 0.35
    ind = np.arange(1,NUM_OBJECTS+1)
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, perch_accs, width, color='g', yerr=perch_stddevs, ecolor='k')
    rects2 = ax.bar(ind + width, cnn_accs, width, color='r', yerr=cnn_stddevs, ecolor='k')

    # Axes labels and title
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Objects')
    ax.set_title(title)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(ind)

    ax.legend((rects1[0], rects2[0]), ('PERCH', 'CNN'))
    plt.savefig(file_name)
    # plt.show()

PlotAccuracy("PERCH vs CNN accuracy - All bins", "overall.pdf", 1, 10)
for bin_count in xrange(1,11):
    print "Computing accuracy for bins with " + str(bin_count) + " items ..."
    plot_title = "#Items in Bin: " + str(bin_count);
    PlotAccuracy(plot_title, "bincount_" + str(bin_count) + ".pdf", bin_count, bin_count)

PlotAccuracy("PERCH vs CNN accuracy - Bin count < 5", "lessthan_5.pdf", 1, 5)
PlotAccuracy("PERCH vs CNN accuracy - Bin count >= 5", "greaterorequal_5.pdf", 6, 10)
