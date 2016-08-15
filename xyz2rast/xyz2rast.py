# -*- coding: utf-8 -*-

__version__ = "0.1.0"


import sys, ntpath, glob
from scipy import spatial, linalg
import numpy as np
import pandas as pd
import datetime as dt

# =========================================================

class mbdata:

    def __init__(self, xyzDir, xyzFmt, rastDir, rastFmt, flowAz):

        self.xyzDir = xyzDir
        self.xyzFmt = xyzFmt
        self.rastDir = rastDir
        self.rastFmt = rastFmt
        self.flowAz = flowAz

        xyzPaths = glob.glob('{0}/*.npy'.format(xyzDir))
        self.nt = len(xyzPaths)

        self.data = pd.DataFrame({'xyzPaths': xyzPaths})
        self.data['time'] = None
        self.data['gridded'] = False
        self.data['xmin'] = None
        self.data['xmax'] = None
        self.data['ymin'] = None
        self.data['ymax'] = None
        self.data['zmin'] = None
        self.data['zmax'] = None
        self.data['raster'] = None

        rastPaths = glob.glob('{0}/*.npy'.format(rastDir))
        tGridded = np.empty(len(rastPaths), 'object')
        for i in range(len(rastPaths)):
            fname = ntpath.basename(rastPaths[i])
            tGridded[i] = dt.datetime.strptime(fname, rastFmt)

        self.tGridded = tGridded
        for i in range(self.nt):
            fname = ntpath.basename(xyzPaths[i])
            self.data.set_value(i, 'time', dt.datetime.strptime(fname, xyzFmt))

            xyz = np.load(self.data.ix[i,'xyzPaths'])
            sw_xyz = self.rotateXY(xyz, flowAz)

            self.data.set_value(i, 'xmin', np.nanmin(sw_xyz[:,0]))
            self.data.set_value(i, 'xmax', np.nanmax(sw_xyz[:,0]))
            self.data.set_value(i, 'ymin', np.nanmin(sw_xyz[:,1]))
            self.data.set_value(i, 'ymax', np.nanmax(sw_xyz[:,1]))
            self.data.set_value(i, 'zmin', np.nanmin(sw_xyz[:,2]))
            self.data.set_value(i, 'zmax', np.nanmax(sw_xyz[:,2]))

            if (self.data.ix[i,'time'] == tGridded).any():
                self.data.set_value(i, 'gridded', True)
                fname = dt.datetime.strftime(self.data.ix[i,'time'], rastFmt)
                rast = np.load('{0}/{1}'.format(self.rastDir, fname))
                self.data.set_value(i, 'raster', rast)

        self.xmin = np.min(self.data.xmin)
        self.xmax = np.max(self.data.xmax)
        self.ymin = np.min(self.data.ymin)
        self.ymax = np.max(self.data.ymax)
        self.zmin = np.min(self.data.zmin)
        self.zmax = np.max(self.data.zmax)
        self.origin = rotateXY(np.array([[self.xmin, self.ymin, self.zmin]]), -flowAz)

    # =========================================================
    def rotateXY(self, xyz, angle):
        """ Rotates every point in an xyz array about the origin by 'angle' """
        rot_xyz = np.empty(xyz.shape)
        theta = np.deg2rad(angle)
        x = xyz[:,0]
        y = xyz[:,1]
        rot_xyz[:,0] = x * np.cos(theta) - y * np.sin(theta)
        rot_xyz[:,1] = x * np.sin(theta) + y * np.cos(theta)
        rot_xyz[:,2] = xyz[:,2]
        return rot_xyz

    # =========================================================
    def translate(self, xyz, xbnd, ybnd):
        """ moves data so [xbnd, ybnd] is at origin """
        xyz[:,0] -= xbnd
        xyz[:,1] -= ybnd
        return xyz

    # =========================================================
    def rescaleXY(self, xyz, old_dx, new_dx, old_dy, new_dy):
        """ Transforms x and y coordinates to grid unit intervals.  Can handle
        anisotropic scales """
        xyz[:,0] *= (old_dx/new_dx)
        xyz[:,1] *= (old_dy/new_dy)
        return xyz

    # =========================================================
    def surfPoint(self, xyz, x, y):
        """ Fits a plane to data in format array([[x,y,z]]) """
            # best-fit linear plane
        A = np.c_[xyz[:,0], xyz[:,1], np.ones(xyz.shape[0])]
        C,_,_,_ = linalg.lstsq(A, xyz[:,2])    # coefficients

        # evaluate at x,y
        z = C[0]*x + C[1]*y + C[2]

        return z

    # =========================================================
    def genGrid(self, xmin, xmax, dx, ymin, ymax, dy):
        """ Generates regular coordinate grid """
        nx = int((xmax-xmin) // dx)
        ny = int((ymax-ymin) // dy)
        remx = (xmax-xmin) % dy
        remy = (ymax-ymin) % dy

        xcords = np.linspace(xmin + dx/2, xmax - remx - dx/2, nx)
        ycords = np.linspace(xmin + dy/2, ymax - remy - dy/2, ny)

        np.meshgrid(xcords, ycords)

        return np.meshgrid(xcords, ycords)


    # =========================================================
    def kdtree(self, xyz):
        """ Generates nearest neighbor kdtree for bathymetry and
        indexable elevation data"""
        ztree = xyz[:,2]
        tree = spatial.KDTree(xyz[:, 0:2])
        return tree, ztree


    # =========================================================
    def gridXYZ(self, xyz, xgrid, ygrid, rSearch):
        """ Grids irregular xyz data """
        tree, ztree = self.kdtree(xyz) # Generate nearest neighbor search tree
        ny, nx = xgrid.shape
        rast = np.empty((ny, nx), 'float')
        for i in range(ny):
            for j in range(nx):

                x = xgrid[i,j]
                y = ygrid[i,j]
                d = tree.query_ball_point([x,y],rSearch) # find nearest neighbors

                if len(d) >= 4:
                    xyz_near = xyz[d]

                    try:
                        rast[i,j] = surfPoint(xyz_near, x, y)
                    except:
                        rast[i,j] = np.nan
                else:
                    rast[i,j] = np.nan

        return rast


    # =========================================================
    def grid_single(self, streamwise_oriented_xyz, nr, xmin, xmax, dx, ymin, ymax, dy):
        """ processing function for single xyz file given row of df_init data """
        # Move to minimum upper right quadrant
        xyz = self.translate(streamwise_oriented_xyz, xmin, ymin)
        # Rescale to coordinates of final DEM cells
        scaled_xyz = self.rescaleXY(xyz, 1, dx, 1, dy)
        scaled_xext = (xmax-xmin)/dx
        scaled_yext = (ymax-ymin)/dy
        # Generate unit coordinate grid at origin
        [scaled_xgrid, scaled_ygrid] = self.genGrid(0, scaled_xext, 1, 0, scaled_yext, 1)
        # Specify search radius for interpolation method (nr = number of cell radii)
        scaled_rsearch = np.sqrt(0.5) * nr

        # Calculate raster and return
        return self.gridXYZ(scaled_xyz, scaled_xgrid, scaled_ygrid, scaled_rsearch)

    def grid_data(self, dx, dy, nr):
        """ batch processing function """
        incomplete = self.data.gridded[self.data.gridded == False].index.tolist()

        try:
            print("{0}/{1} complete.  Resuming.".format(incomplete[0], self.nt))
            for i in incomplete:
                streamwise_xyz = self.rotateXY(np.load(self.data.ix[i,'xyzPaths']),
                                          self.flowAz)
                rast = self.grid_single(streamwise_xyz, nr,
                                   self.xmin, self.xmax, dx,
                                   self.ymin, self.ymax, dy)
                self.data.set_value(i, 'raster', rast)
                self.data.ix[i, 'gridded'] = True
                fname = dt.datetime.strftime(self.data.ix[i,'time'], self.rastFmt)
                outfile = '{0}/{1}'.format(self.rastDir, fname)
                np.save(outfile,rast)
                print("({0}/{1}) completed at {2})".format(i+1, self.nt, dt.datetime.now()))
        except:
            print("Gridding Complete")

        Z = np.empty((self.nt, self.data.raster[0].shape[0],
                      self.data.raster[0].shape[1]), 'float')
        for t in range(self.nt):
            Z[t] = self.data.raster[t]

        return Z, self.data.time, self.dx, self.dy, self.origin

#  =============================================================================

def load_data(xyzDir, xyzFmt, rastDir, rastFmt, flowAz):
    return mbdata(xyzDir, xyzFmt, rastDir, rastFmt, flowAz)
