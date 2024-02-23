import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from constants import DataFile
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from gudhi.representations import PersistenceImage


import seaborn as sns
import matplotlib.pyplot as plt

class RBCPD(object):

    def __init__(self, pd_filename):

        self.filename = pd_filename
        self.name = pd_filename.name

    def load_pd(self):
        pd_file = np.loadtxt(str(self.filename))
        # Loop through the pd file to find different measure
        pd_points = {}
        b = -1
        for p in pd_file:
            if p[0] == -1:
                # This indicates the start of a new dimension
                b = int(round(p[1]))
                pd_points['betti_' + str(b)] = []
            else:
                # We have a persistence point which we need to save
                # Remove trivial circles
                if b == 1 and np.array(p)[0] < 8 and np.array(p)[1]<8:
                    continue
                else:
                    pd_points['betti_' + str(b)].append(np.array(p))
        for b in [1, 2]:
            if len(pd_points['betti_' + str(b)]) == 0:
                pd_points['betti_' + str(b)].append(np.array([0, 0]))

        return pd_points

    def compute_statistical_summaries(self):
        # Compute persistence summaries of pd points.
        pd_points = self.load_pd()
        features = {}
        for betti_b, points in pd_points.items():
            summaries = PDsummaries(points).get_summaries()
            for measure, statistics in summaries.items():
                for statistic, value in statistics.items():
                    features[betti_b + '_' + measure + '_' + statistic] = value

        return features

    def vectorize(self):
        features = {}
        pd_points = self.load_pd()
        persistent_volume = np.max([p[1] for p in pd_points['betti_2']])
        features['persistent_volume'] = persistent_volume
        pd_b1_births = [p[0] for p in pd_points['betti_1']]
        n_b1_births = 1
        b1_births = [0 for _ in range(n_b1_births)]
        persistent_compressibility = [0 for _ in range(n_b1_births)]
        for n, pb in enumerate(pd_b1_births):
            if n < n_b1_births:
                b1_births[n] = pd_b1_births[-1-n]
                persistent_compressibility[n] = b1_births[n] / persistent_volume
        for n in range(n_b1_births):
            features['b1_birth_' + str(n)] = b1_births[n]
            features['persistent_compressibility_' + str(n)] = persistent_compressibility[n]

        return features


    def computer_persistence_image(self):
        # Compute persistence summaries of pd points.
        pd_points = self.load_pd()
        res = 50
        features = {}
        for betti_b, points in pd_points.items():
            PI = PersistenceImage(bandwidth=20, resolution=[res, res], im_range=[0, 60, 0, 60])
            im = PI.fit_transform([np.array(points)])
            for i in range(res ** 2):
                features[betti_b + '_' + str(i)] = im[0][i]

        return features

class PDsummaries(object):

    def __init__(self, pd_points):
        self.pd_points = pd_points
        self.measure = {'births': [p[0] for p in pd_points],
                         'deaths': [p[1] for p in pd_points],
                         'midpoints': [(p[0] + p[1]) / 2 for p in pd_points],
                         'lifespans': [p[1] - p[0] for p in pd_points]}

        self.n_summaries = len(self.get_summaries())

    def get_summaries(self):
        summaries = {}
        for statistic, values in self.measure.items():
            summary = {'mean': np.mean(values),
                     'sdev': np.std(values),
                     'iqr': np.percentile(values, 75) - np.percentile(values, 25),
                     'range': np.max(values) - np.min(values),
                     '10pc': np.percentile(values, 10),
                     '25pc': np.percentile(values, 25),
                     '75pc': np.percentile(values, 75),
                     '90pc': np.percentile(values, 90),
                     'max': np.max(values),
                     'min': np.min(values)}
            summaries[statistic] = summary

        return summaries

class LearnRBC(object):

    def __init__(self, raw_data_dir, data_dir, results_dir, by_class=True):
        self.raw_data_dir = raw_data_dir
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.by_class = by_class

        if self.by_class:
            files = {}
            classes = []
            for c in list(self.raw_data_dir.iterdir()):
                files[c.name] = []
                classes.append(c.name)
                for f in list(c.iterdir()):
                    if f.name[-4:] == '.tif':
                        files[c.name].append(f.name)
            self.files = files
            self.classes = classes

    def save_dat_data(self):
        if self.by_class:
            (self.results_dir / 'training').mkdir(exist_ok=True)
            for c in self.classes:
                (self.results_dir / 'training' / c).mkdir(exist_ok=True)
                for f in self.files[c]:
                    feature_vector = RBCPD(self.data_dir / c / (f + '_pd.csv')).compute_statistical_summaries()
                    pd.DataFrame({str(len(feature_vector)) + ' ASCII':
                                      [' ' + ' '.join('%.6f' %x for x in list(feature_vector.values()))]})\
                        .to_csv(self.results_dir / 'training' / c / (f[:-4] + '.dat'), index=None)


    def save_data(self):
        D = []
        if self.by_class:
            for c in self.classes:
                for f in self.files[c]:
                    d = {'number': f[:-4],
                         'class': c,
                         'SDE': get_sde(c),
                         'combined_class': get_combined_class(c),
                         }
                    rbcpd = RBCPD(self.data_dir / c / (f + '_pd.csv'))
                    # Try using the statistical summaries
                    features = rbcpd.compute_statistical_summaries()
                    # Try using a more bespoke vectorisation
                    # features = rbcpd.vectorize()
                    # Try getting persistence image
                    # features = rbcpd.computer_persistence_image()
                    d.update(features)

                    D.append(d)
        df = pd.DataFrame.from_dict(D)
        df.to_csv(self.results_dir / 'data.csv')

    def import_dat_data(self):
        D = []
        if self.by_class:
            for c in self.classes:
                for f in self.files[c]:
                    d = {'number': f[:-4],
                         'class': c,
                         'SDE': get_sde(c),
                         'combined_class': get_combined_class(c),
                         }
                    dat_file = np.genfromtxt(self.data_dir / c / (f[:-4] + '.dat'), skip_header=1)
                    for di, dat in enumerate(dat_file):
                        d['dat_' + str(di)] = dat

                    D.append(d)
        df = pd.DataFrame.from_dict(D)
        df.to_csv(self.results_dir / 'data.csv')
    def load_data(self):

        D = pd.read_csv(self.results_dir / 'data.csv', index_col=0)
        return D

    def learn(self):

        D = self.load_data()
        c = list(D.columns)
        features = [f for f in c if f not in ['class', 'SDE', 'number', 'combined_class']]
        X = D[features]

        # First do classification
        t = 'classify_'
        y = D['combined_class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Random Forests
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        sc = clf.score(X_test, y_test)
        cm = confusion_matrix(y_test, clf.predict(X_test), normalize=None)
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        fig, ax = plt.subplots()
        cm_disp.plot(ax=ax, xticks_rotation='vertical')
        fig.savefig(self.results_dir / (t + 'rf_score_' + str(sc) + '.png'), bbox_inches='tight')

        # Multi-layer perceptron
        mlp = MLPClassifier()
        mlp.fit(X_train, y_train)
        sc = mlp.score(X_test, y_test)
        cm = confusion_matrix(y_test, mlp.predict(X_test), normalize=None)
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
        fig, ax = plt.subplots()
        cm_disp.plot(ax=ax, xticks_rotation='vertical')
        fig.savefig(self.results_dir / (t + 'mlp_score_' + str(sc) + '.png'), bbox_inches='tight')


        # Now do regression on the SDE pathway
        t = 'regress_'
        D_sde = D.dropna()
        D_sde = D_sde.sort_values('SDE')
        X = D_sde[features]
        y = D_sde['SDE'].astype(str)
        labels = ['-1.0', '-0.67', '-0.33', '0.0', '0.33', '0.67', '1.0']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Random Forests
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        sc = clf.score(X_test, y_test)
        cm = confusion_matrix(y_test, clf.predict(X_test), labels=labels,  normalize=None)
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        fig, ax = plt.subplots()
        cm_disp.plot(ax=ax, xticks_rotation='vertical')
        fig.savefig(self.results_dir / (t + 'rf_score_' + str(sc) + '.png'), bbox_inches='tight')

        # Multi-layer perceptron
        mlp = MLPClassifier()
        mlp.fit(X_train, y_train)
        sc = mlp.score(X_test, y_test)
        cm = confusion_matrix(y_test, mlp.predict(X_test), labels=labels, normalize=None)
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
        fig, ax = plt.subplots()
        cm_disp.plot(ax=ax, xticks_rotation='vertical')
        fig.savefig(self.results_dir / (t + 'mlp_score_' + str(sc) + '.png'), bbox_inches='tight')



    def save_magic_key(self):
        D = []
        if self.by_class:
            for c in self.classes:
                for f in self.files[c]:
                    rbcpd = RBCPD(self.data_dir / c / (f + '_pd.csv'))
                    features = rbcpd.compute_statistical_summaries()
                    D.append({'class': c, 'number': f[:-4], 'thresh': features['betti_2_deaths_max']})
        df = pd.DataFrame(D)
        df.to_csv(self.results_dir / 'magic_key.csv')




def get_sde(c):

    if not (c[0].isnumeric() or c[1].isnumeric()):
        sde = np.nan
    else:
        sde = []
        for s in c:
            if s == '-' or s.isnumeric() or s == '.':
                sde.append(s)
            else:
                break
        sde = float(''.join(sde))
    return sde


def get_combined_class(c):
    sde = get_sde(c)
    if not np.isnan(sde):
        return 'SDE'
    else:
        return c

# class Labels(object):
#
#     def __init__(self, files):
#
#         self.files = files
#
#     def get_class_divisions(self):
#
#         names = [file.name for file in self.files]
#         class_counts = {}
#         for name in names:
#             name_class = get_class(name)
#             if name_class in class_counts.keys():
#                 class_counts[name_class] += 1
#             else:
#                 class_counts[name_class] = 1
#         pd.DataFrame.from_dict(class_counts, orient="index").to_csv('./class_counts.csv')
#         return class_counts



if True or __name__=='main':

    # # My analysis
    data_name = 'cytoShape_dataset'
    raw_data_dir = Path('/scratch/mcdonald/cytoShapeNet/all_data_download/data/training')
    data_dir = Path('/home/mcdonald/Documents/CSElab/rbc_phenotyping/data/cytoShape_dataset_50t_1500p')
    results_dir = Path('/home/mcdonald/Documents/CSElab/rbc_phenotyping/results/combined_classes/')

    lrbc = LearnRBC(raw_data_dir, data_dir, results_dir)
    # lrbc.save_magic_key()
    lrbc.save_data()
    lrbc.learn()

    # Their analysis
    # data_name = 'cytoShape_dataset'
    # raw_data_dir = Path('/scratch/mcdonald/cytoShapeNet/all_data_download/data/training')
    # data_dir = Path('/home/mcdonald/Documents/CSElab/rbc_phenotyping/data/dat/')
    # results_dir = Path('/home/mcdonald/Documents/CSElab/rbc_phenotyping/results/dat/')
    #
    # lrbc = LearnRBC(raw_data_dir, data_dir, results_dir)
    # # lrbc.save_magic_key()
    # lrbc.import_dat_data()
    # lrbc.learn()








