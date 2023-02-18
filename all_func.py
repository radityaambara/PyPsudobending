from scipy.spatial import KDTree,Voronoi,ConvexHull, Delaunay
import numpy as np
from math import sqrt

"""This is class for velocity grid:
Here, you can create a velocity grid object and use the ray tracing method 
to determine the length of the ray path within each grid. 
For calculating the ray path length using tomography inversion, 
you can input irregular nodes. However, for ray tracing, 
you must input rectangular and regular grid nodes.
"""
class VelocityGrid:
    """
    input velocity grid node (x, y, z)
    xnode=1D array of regular grid x
    ynode=1D array of regular grid y
    znode=1D array of regular grid z
    gridV=3D array of regular grid velocity, you can do the interpolation first to have grid V from
    irreguler xyz velocity data into reguler xnode,ynode,znode grid.

    deltn=space between regular node
    delt=maximum distance in each seismic ray
    iter1 & iter2 = iteration number for psudo-bending
    tmin = error minimum for psudo-bending
    """
    def __init__(self,node,xnode,ynode,znode,gridV,deltn,delt,xfac,iter1,iter2,tmin):
        self.node=node
        self.xnode=xnode
        self.ynode=ynode
        self.znode=znode
        self.gridV=gridV
        self.deltn=deltn
        self.delt=delt
        self.xmin=np.amin(xnode)
        self.ymin=np.amin(ynode)
        self.zmin=np.amin(znode)
        self.xfac=xfac
        self.iter1=iter1
        self.iter2=iter2
        self.tmin=tmin

    def vel(self,x, y, z):
        ix = int((x - self.xmin) / self.deltn)
        iy = int((y - self.ymin) / self.deltn)
        iz = int((z - self.zmin) / self.deltn)
        ix1 = ix + 1
        iy1 = iy + 1
        iz1 = iz + 1

        xf = (x - self.xnode[ix]) / (self.xnode[ix1] - self.xnode[ix])
        yf = (y - self.ynode[iy]) / (self.ynode[iy1] - self.ynode[iy])
        zf = (z - self.znode[iz]) / (self.znode[iz1] - self.znode[iz])
        xf1 = 1 - xf
        yf1 = 1 - yf
        zf1 = 1 - zf

        w1 = xf1 * yf1 * zf1
        w2 = xf * yf1 * zf1
        w3 = xf1 * yf * zf1
        w4 = xf * yf * zf1
        w5 = xf1 * yf1 * zf
        w6 = xf * yf1 * zf
        w7 = xf1 * yf * zf
        w8 = xf * yf * zf
        v = (w1 * self.gridV[ix, iy, iz]) + (w2 * self.gridV[ix1, iy, iz]) + (w3 * self.gridV[ix, iy1, iz]) + (
                w4 * self.gridV[ix1, iy1, iz]) + (w5 * self.gridV[ix, iy, iz1]) \
            + (w6 * self.gridV[ix1, iy, iz1]) + (w7 * self.gridV[ix, iy1, iz1]) + (w8 * self.gridV[ix1, iy1, iz1])
        return v

    def veld(self,x, y, z):
        ix = int((x - self.xmin) / self.deltn)
        iy = int((y - self.ymin) / self.deltn)
        iz = int((z - self.zmin) / self.deltn)
        ix1 = ix + 1
        iy1 = iy + 1
        iz1 = iz + 1

        xf = (x - self.xnode[ix]) / (self.xnode[ix1] - self.xnode[ix])
        yf = (y - self.ynode[iy]) / (self.ynode[iy1] - self.ynode[iy])
        zf = (z - self.znode[iz]) / (self.znode[iz1] - self.znode[iz])
        xf1 = 1 - xf
        yf1 = 1 - yf
        zf1 = 1 - zf

        dvx = ((yf1 * zf1 * (self.gridV[ix1, iy, iz] - self.gridV[ix, iy, iz])) + (
                yf * zf1 * (self.gridV[ix1, iy1, iz] - self.gridV[ix, iy1, iz])) + (
                       yf1 * zf * (self.gridV[ix1, iy, iz1] - self.gridV[ix, iy, iz1])) + (
                       yf * zf * (self.gridV[ix1, iy1, iz1] - self.gridV[ix, iy1, iz1]))) / (self.xnode[ix1] - self.xnode[ix])

        dvy = ((xf1 * zf1 * (self.gridV[ix, iy1, iz] - self.gridV[ix, iy, iz])) + (
                xf * zf1 * (self.gridV[ix1, iy1, iz] - self.gridV[ix1, iy, iz])) + (
                       xf1 * zf * (self.gridV[ix, iy1, iz1] - self.gridV[ix, iy, iz1])) + (
                       xf * zf * (self.gridV[ix1, iy1, iz1] - self.gridV[ix1, iy, iz1]))) / (self.ynode[iy1] - self.ynode[iy])

        dvz = ((xf1 * yf1 * (self.gridV[ix, iy, iz1] - self.gridV[ix, iy, iz])) + (
                xf * yf1 * (self.gridV[ix1, iy, iz1] - self.gridV[ix1, iy, iz])) + (
                       xf1 * yf * (self.gridV[ix, iy1, iz1] - self.gridV[ix, iy1, iz])) + (
                       xf * yf * (self.gridV[ix1, iy1, iz1] - self.gridV[ix1, iy1, iz]))) / (self.znode[iz1] - self.znode[iz])
        return dvx, dvy, dvz

    def tt(self,path):
        tt = 0
        for j in range(0, path.shape[0] - 1):
            jarak = sqrt(np.sum((path[j] - path[j + 1]) ** 2, axis=0))
            space = int(jarak / self.delt)
            if space < 3:
                space = 3
            jarak2 = jarak / (space - 1)
            xinc = np.linspace(path[j][0], path[j + 1][0], space)
            yinc = np.linspace(path[j][1], path[j + 1][1], space)
            zinc = np.linspace(path[j][2], path[j + 1][2], space)
            s1 = self.vel(xinc[0], yinc[0], zinc[0])
            for i in range(1, space):
                s2 = self.vel(xinc[i], yinc[i], zinc[i])
                tt += jarak2/(s2 + s1)
                s1 = np.copy(s2)
        return tt * 2

    def pertub(self,a, b, c):
        mid = ((b - a) / 2) + a
        gVx, gVy, gVz = self.veld(mid[0], mid[1], mid[2])
        gV = np.array([gVx, gVy, gVz])
        # arahn
        n = gV - (np.dot(gV, (b - a)) * (b - a) / (np.dot(b - a, b - a)))
        if not np.any(n) == False:
            n = n / sqrt(np.dot(n, n))
            Vmid = self.vel(mid[0], mid[1], mid[2])
            Sa = 1 / self.vel(a[0],a[1],a[2])
            Sb = 1 / self.vel(b[0],b[1],b[2])
            # penentuan R
            kL=sqrt(np.dot(b-mid,b-mid))
            kC=(Sa+Sb)/2

            R=(-(kC*Vmid+1)/(np.dot(4*kC*n,gV)))+sqrt((((kC*Vmid)+1)**2/((np.dot(4*kC*n,gV))**2))+(kL**2/2/kC/Vmid))
#            rcur = Vmid / sqrt(np.dot(n, n))
#            if rcur ** 2 > 0.25 * (np.dot(b - a, b - a)):
#                R = rcur - sqrt(rcur ** 2 - 0.25 * (np.dot(b - a, b - a)))
#            else:
#               R = rcur

            new = mid + (n * R)
            c = (self.xfac * (new - c)) + c
        return c

    def doublepath(self,path):
        it = path.shape[0] - 1
        pathnew = path[0]
        for j in range(0, it):
            dis = sqrt(np.sum((path[j + 1] - path[j]) ** 2, axis=0))
            if dis < (2 * self.delt):
                pathnew = np.copy(path)
                break
            c = ((path[j + 1] - path[j]) / 2) + path[j]
            pathnew = np.vstack((pathnew, c, path[j + 1]))
        return pathnew

    def pertubray(self,path):
        it = int((path.shape[0] - 1) / 2)
        le = (path.shape[0] - 1)
        pathn = np.copy(path)
        for i in range(1, it + 1):
            pathn[i, :] = self.pertub(pathn[i - 1], pathn[i + 1], pathn[i])
            if i != it:
                pathn[le - i, :] = self.pertub(pathn[le - i - 1], pathn[le - i + 1], pathn[le - i])
        return pathn

    """method for psudobending"""
    def psudobending(self,path):
        pathn = self.doublepath(path)
        tt0 = self.tt(pathn)
        for i in range(0, self.iter1):
            for j in range(0, self.iter2):
                #print(i,j)
                pathn1 = self.pertubray(pathn)
                tt1 = self.tt(pathn1)
                if abs(tt0 - tt1) <= self.tmin:
                    break
                tt0 = np.copy(tt1)
                pathn = np.copy(pathn1)
            pathn = self.doublepath(pathn)
            pathn1 = self.pertubray(pathn)
            tt1 = self.tt(pathn1)
            if abs(tt0 - tt1) <= self.tmin:
                break
            tt0 = np.copy(tt1)
            pathn = np.copy(pathn1)
        return pathn, tt0

    """method to count raypath length in each irregular node"""
    def crkernel(self, path):
        mytree=KDTree(self.node)
        krn = np.zeros((1, len(self.node)))
        for j in range(0, path.shape[0] - 1):
            jarak = sqrt(np.sum((path[j] - path[j + 1]) ** 2, axis=0))
            space = int(round((jarak / self.delt)))
            if space < 3:
                space = 3
            jarak2 = jarak / (space - 1)
            xinc = np.linspace(path[j][0], path[j + 1][0], space)
            yinc = np.linspace(path[j][1], path[j + 1][1], space)
            zinc = np.linspace(path[j][2], path[j + 1][2], space)
            p1 = np.array([xinc[0], yinc[0], zinc[0]])
            ind1 = mytree.query(p1)[1]
            for i in range(1, space):
                p2 = np.array([xinc[i], yinc[i], zinc[i]])
                ind2 = mytree.query(p2)[1]
                if ind1 != ind2:
                    jarak3=jarak2/10
                    xinc_h=np.linspace(xinc[i-1],xinc[i], 11)
                    yinc_h=np.linspace(yinc[i-1],yinc[i], 11)
                    zinc_h=np.linspace(zinc[i-1],zinc[i], 11)
                    for h in range(1,11):
                        p2=np.array([xinc_h[h], yinc_h[h], zinc_h[h]])
                        ind2 = mytree.query(p2)[1]
                        if ind1 != ind2:
                            krn[0, int(ind1)] += jarak3 / 2
                            krn[0, int(ind2)] += jarak3 / 2
                        else:
                            krn[0, int(ind1)] += jarak3
                        ind1=np.copy(ind2)
                else:
                    krn[0, int(ind1)] += jarak2
                ind1 = np.copy(ind2)
        return krn

    """method for create hypocenter derivative"""
    def hypo_deriv(self, path):
        s = 1 / self.vel(path[0][0], path[0][1], path[0][2])
        delta_s = path[1]-path[0]
        delta_s_abs = sqrt(np.dot(delta_s,delta_s))
        hypo_der=-s*delta_s/delta_s_abs
        hypo_der=np.insert(hypo_der,0,1)
        return hypo_der

    """method to call all the three method for each data"""
    def forwardtwopoints(self, path):
        pathn, tt = self.psudobending(path)
        gkernl = self.crkernel(pathn)
        hypo_der= self.hypo_deriv(pathn)
        return pathn, tt, gkernl, hypo_der

    """method to change node if there is a change"""
    def change_node(self, newnode):
        self.node=newnode

"""function to count the volume for each voronoi diagram for a given xyz irregular node"""
def vor_volumes(points):
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = 0
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    return vol

def check_dis(p1,p2,p3,p4,p5,deltn):
    dis1=np.sqrt(np.dot((p1-p2),(p1-p2)))
    dis2=np.sqrt(np.dot((p1-p3),(p1-p3)))
    dis3=np.sqrt(np.dot((p1-p4),(p1-p4)))
    dis4=np.sqrt(np.dot((p1-p5),(p1-p5)))
    if dis1>=deltn and dis2>=deltn and dis3>=deltn and dis4>=deltn:
        dis=True
    else:
        dis=False
    return dis

"""function for creating spatial smoothing matrix based on triangulation neighboar"""
def smooth_matrix(node_inv):
    def find_neighbors(pindex,triang):
        return triang.vertex_neighbor_vertices[1][
               triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex + 1]]

    tri=Delaunay(node_inv)
    mat_list=[]
    for i in range(len(node_inv)):
        neighbor_indices=find_neighbors(i,tri)
        for j in range(len(neighbor_indices)):
            mat_j=np.zeros((len(node_inv)))
            mat_j[i]=1
            mat_j[neighbor_indices[j]]=-1
            mat_list.append(mat_j)

    return np.array(mat_list)

"""class for checking add node"""
class is_addnode:
    def __init__(self,hit_count,tres,node,deltn,interpP,interpS):
        self.hit_count=hit_count
        self.tres=tres
        self.node=node
        self.deltn=deltn
        self.interpP=interpP
        self.interpS=interpS
    def cek_dis(self,p0,p1,p2,p3,p4):
        dis1 = np.sqrt(np.dot((p0 - p1), (p0 - p1)))
        dis2 = np.sqrt(np.dot((p0 - p2), (p0 - p2)))
        dis3 = np.sqrt(np.dot((p0 - p3), (p0 - p3)))
        dis4 = np.sqrt(np.dot((p0 - p4), (p0 - p4)))
        if dis1 >= 2*self.deltn and dis2 >= 2*self.deltn and dis3 >= 2*self.deltn and dis4 >= 2*self.deltn:
            dis = True
        else:
            dis = False
        return dis

    """add node if condition is met"""
    def cek_add(self,sim):
        p1=sim[0]
        p2=sim[1]
        p3=sim[2]
        p4=sim[3]
        cek_sum=self.hit_count[p1]+self.hit_count[p2]+self.hit_count[p3]+self.hit_count[p4]
        added_node=np.array([-99])
        add_velP=np.array([-99])
        add_velS=np.array([-99])
        if cek_sum>self.tres:
            added_node=(self.node[p1,:]+self.node[p2,:]+self.node[p3,:]+self.node[p4,:])/4
            check_added_node=self.cek_dis(added_node,self.node[p1,:],self.node[p2,:],self.node[p3,:],self.node[p4,:])
            if check_added_node==True:
                add_velP=self.interpP(added_node[0],added_node[1],added_node[2])
                add_velS=self.interpS(added_node[0],added_node[1],added_node[2])

        return added_node,add_velP,add_velS




