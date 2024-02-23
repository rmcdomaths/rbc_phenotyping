import os
import numpy as np
import pandas as pd
from skimage import measure
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
import point_cloud_utils as pcu
from ripser import ripser
from persim import plot_diagrams
from multiprocessing import Pool
from scipy.spatial.distance import cdist
import time
import re

class SlicedRBC(object):

    def __init__(self, filename, save_dir, save_by_class):

        self.filename = filename
        self.name = filename.name
        self.phenotype = filename.parent.name
        self.save_dir = save_dir
        self.save_by_class = save_by_class
        if save_by_class:
            self.save_to = self.save_dir / self.phenotype
        else:
            self.save_to = self.save_dir
        magic_key_file = pd.read_csv(save_dir / 'magic_key.csv', index_col=0)
        magic_key = {}
        for i in range(len(magic_key_file.index)):
            c = magic_key_file.iloc[i]['class']
            if c not in magic_key.keys():
                magic_key[c] = {}
            magic_key[c][str(magic_key_file.iloc[i]['number'])] = magic_key_file.iloc[i]['thresh']
        self.magic_key = magic_key


    def to_numpy(self):
        # Convert tif file to numpy
        return np.array(io.imread(self.filename))

    def do_marching_cubes(self, level=50):
        # Convert 3d np array to some surface format
        A = self.to_numpy()
        # Transpose so that vertical dimension is the final one
        shape = np.array(np.shape(A))
        ordering = (1, 2, 0)
        debug = False
        if debug:
            # Snippet to check if the thing is really spherical
            z_range = np.where(np.max(np.max(A, axis=-1), axis=-1) > level)
            z_min, z_max = np.min(z_range), np.max(z_range)
            x_min = np.min([np.min(np.where(a > level)[0]) for a in [A[i] for i in range(np.shape(A)[0])] if
                            np.any(np.where(a > level)[0])])
            x_max = np.max([np.max(np.where(a > level)[0]) for a in [A[i] for i in range(np.shape(A)[0])] if
                            np.any(np.where(a > level)[0])])
            y_min = np.min([np.min(np.where(a > level)[1]) for a in [A[i] for i in range(np.shape(A)[0])] if
                            np.any(np.where(a > level)[1])])
            y_max = np.max([np.max(np.where(a > level)[1]) for a in [A[i] for i in range(np.shape(A)[0])] if
                            np.any(np.where(a > level)[1])])
        verts, faces, normals, values = measure.marching_cubes(np.transpose(A, ordering),
                                                               level=50,
                                                               spacing=(1, 1, .75))
        if debug:
            px_min, px_max = np.min(verts[0, :]), np.max(verts[0, :])
            py_min, py_max = np.min(verts[1, :]), np.max(verts[1, :])
            pz_min, pz_max = np.min(verts[2, :]), np.max(verts[2, :])

        return verts, faces, normals, values

    def get_pointcloud(self, target_points):
        # Get pointcloud - just the vertices, or fewer if you want to sample from the surface
        verts, faces, normals, values = self.do_marching_cubes()

        if not target_points:
            pointcloud = verts
        else:
            fid, bc = pcu.sample_mesh_poisson_disk(verts, faces, num_samples=target_points)
            rand_positions = pcu.interpolate_barycentric_coords(faces, fid, bc, verts)
            rand_normals = pcu.interpolate_barycentric_coords(faces, fid, bc, normals)
            pointcloud = rand_positions

        debug = False
        if debug:
            verts = self.get_pointcloud(target_points=1000)
            D = cdist(verts, verts)
            cidx = np.where(np.abs(D - np.max(D)) < 1)
            max_v = [verts[v, :] for v in cidx[0]] + [verts[v, :] for v in cidx[1]]
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=1, alpha=.5, color='b')
            [ax.scatter(v[0], v[1], v[2], s=10, color='r') for v in max_v]
            fig.savefig('./sphere.png')
            plt.close(fig)

        return pointcloud

    def save_surface(self):
        # Plot the surface
        verts, faces, normals, values = self.do_marching_cubes()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces)
        fig.savefig(self.save_to / (self.name + '_mesh.png'))
        plt.close()

        points = self.get_pointcloud(target_points=1500)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter([points[:, 0]], [points[:, 1]], [points[:, 2]])
        fig.savefig(self.save_to / (self.name + '_points.png'))
        plt.close()

        return None

    def compute_persistence(self, n_points=1500, maxdim=2):
        points = self.get_pointcloud(target_points=n_points)
        D = cdist(points, points, 'euclidean')
        D_ax = np.max(np.max(D))
        # In a sphere of radius r, max(betti_2(d))=r*sqrt(8/3), ie the void will be filled in at VR(eps) for this eps
        # Therefore this is the maximum threshold we should consider
        # Note that the maximum threshold will be smaller if the rbc is not a perfect sphere
        # To have max(betti_2(d)) = D_ax, we could potentially use a different complex? TBI.
        # self.threshold = (D_ax * np.sqrt(8/3))
        self.threshold = 2 + round(self.magic_key[self.phenotype][re.sub("^0+(?!$)", "", self.name[:-4])])

        diagrams = ripser((np.array(points)), thresh=self.threshold, maxdim=maxdim)

        return diagrams

    def save(self, save_surface=True, plot_diagram=True, save_diagram=20):

        if save_surface:
            self.save_surface()
        if plot_diagram or save_diagram:
            d = self.compute_persistence()
            if plot_diagram:
                fig, ax = plt.subplots()
                plot_diagrams(d['dgms'], show=False, ax=ax)
                fig.savefig(self.save_to / (self.name + '_pd_plot.png'))
                plt.close()
            if save_diagram:
                persistences = [p[:, 1] - p[:, 0] for p in d['dgms']]
                # Just take the points of top n persistence
                n = save_diagram
                top_n_idx_dim1 = np.argsort(persistences[1])[-n:]
                top_n_idx_dim2 = np.argsort(persistences[2])[-n:]
                pre_points_to_save = [np.array([d['dgms'][1][i] for i in top_n_idx_dim1]),
                                      np.array([d['dgms'][2][i] for i in top_n_idx_dim2])]

                # Convert any points of infinite persistence to the max threshold
                points_to_save = []
                for dim in pre_points_to_save:
                    p_dim = []
                    for p in dim:
                        if p[1] == np.inf:
                            p_dim.append(np.array([p[0], self.threshold]))
                        else:
                            p_dim.append(p)
                    points_to_save.append(p_dim)
                # Save to csv
                np.savetxt(self.save_to / (self.name +  '_pd.csv'),
                           np.vstack([np.array([-1, 1])] + points_to_save[0] + [np.array([-1, 2])] + points_to_save[1]))


def parallel_save(file):

    save_dir = Path('/home/mcdonald/Documents/CSElab/rbc_phenotyping/data/cytoShape_dataset_50t_1500p/')
    srbc = SlicedRBC(file, save_dir=save_dir, save_by_class=True)
    srbc.save(save_diagram=10000, plot_diagram=True)
    return None


if __name__ == '__main__':

    data_dir = Path('/scratch/mcdonald/cytoShapeNet/all_data_download/data/training/')
    save_dir = Path('/home/mcdonald/Documents/CSElab/rbc_phenotyping/data/cytoShape_dataset_50t_1500p/')

    all_folders = list(data_dir.iterdir())
    all_classes = [f.name for f in all_folders]
    all_classes.reverse()
    for c in all_classes:

        print('Computing class: ' + c)

        (save_dir / c).mkdir(exist_ok=True)
        all_files = list((data_dir / c).iterdir())
        all_files = [f for f in all_files if '.tif' in f.name]
        save_by_class = True
        st = time.time()
        with Pool(processes=13) as pool:
            pool.map(parallel_save, all_files)
        et = time.time()
        print('Time taken: ', (et - st) // 60, ' minutes ', (et - st) % 60, ' seconds')

    # data_dir = Path('/scratch/mcdonald/SHAPR_torch/data/dataset_for_3D_reconstruction/obj/')
    # all_files = list(data_dir.iterdir())
    # with Pool() as pool:
    #     pool.map(parallel_save, all_files)




