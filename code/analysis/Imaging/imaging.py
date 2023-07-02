__version__ = "2023.06.26-1"

import numpy, pandas, pymc, scipy, sympy, ROOT, uproot, geometry, time, itertools, dask
from tqdm.autonotebook import tqdm
import numpy.random as rd
import matplotlib.pyplot as plt
tqdm.pandas()

def soe(df, cube, random=False, radius=15, sampling=1000, iterations=1000):
    '''
    #  9.7s /(   500it x 10000samp x 100cones ) =  20ns/(it x samp x cones)
    # 29.9s /(  1000it x  1000samp x 200cones ) = 150ns/(it x samp x cones)
    # 61.3s /(  1000it x  1000samp x 500cones ) = 120ns/(it x samp x cones)
    Returns the possible position of emission for each detection.

            Parameters:
                    df (pandas.DataFrame[Cone]): A decimal integer
                    cube (pandas.Series[CubeFOV]): Another decimal integer
                    radius (int): Radius of the sphere for the density calculation
                    sampling (int): Number of possible interaction positions per cone, per iteration
                    iterations (int): Number of iterations before calculating the last position
                    #rr (int): Number of resolution recovery samplings for each measurement
                    
            Returns:
                    binary_sum (pandas.Series[Point3]): Possible position of emission.
    '''
    start = time.time()
    PRNGs=[rd.Generator(rd.MT19937(s)) for s in rd.SeedSequence(20000609).spawn(3)]

    # Generate cones
    df["cones"] = df.progress_apply(
        lambda entry: geometry.Cone(
            geometry.Point3(
                entry.X_s,
                entry.Y_s,
                entry.Z_s
            ),
            geometry.Point3(
                entry.X_s,
                entry.Y_s,
                entry.Z_s
            )-geometry.Point3(
                entry.X_a,
                entry.Y_a,
                entry.Z_a
            ),
            geometry.Angle(numpy.rad2deg(numpy.arccos(entry.costheta)))
        ),
        axis=1
    )
    print(f"Geometry created: {time.time()-start:.2f}s")

    # Generate points in cone's surface
    #points = pandas.DataFrame(itertools.product(df.index,range(iterations)),columns=["idx","it"])
    points = pandas.DataFrame(itertools.product(df.index,range(iterations),range(sampling)),columns=["idx","it","sample"])
    points["cone"]   = points.idx.apply(lambda idx: df.loc[idx, "cones"])
    points["id"]     = points.idx.apply(lambda idx: df.loc[idx, "id"])
    points["rand0"]  = PRNGs[0].random(points.shape[0])
    points["rand1"]  = PRNGs[1].random(points.shape[0])
    points["choice"] = PRNGs[2].random(points.shape[0])
    points["point"]  = points.progress_apply(lambda entry: entry.cone(entry.rand0, entry.rand1),axis=1)
    t0 = points.shape[0]
    points = points[points.point.apply(cube.inside)]
    print(f"FOV efficiency: {points.shape[0]/t0*100}%")
    print(f"Possible positions created: {time.time()-start:.2f}s")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    l=pandas.DataFrame({"x":[],"y":[],"z":[]})
    l["x"] = points.point.apply(lambda entry: entry.get_x())
    l["y"] = points.point.apply(lambda entry: entry.get_y())
    l["z"] = points.point.apply(lambda entry: entry.get_z())
    l = l[:10000]
    ax.scatter(l.x, l.y, l.z,color="b", alpha=0.2)
    plt.show()

    # Generate a sphere around each point
    points["sphere"]  = points.point.apply(lambda p: geometry.Sphere(p, radius))

    # Each point starts with a False status and the statues is set to True according to the acceptance probability
    points["status"]  = False
    points["density"] = 0
    # Start with possibility "True" for every point in first iteration
    points.loc[points["it"] == 0, "status"]  = True
    points.loc[points["it"] == 0, "choice"]  = 1
    points.loc[points["it"] == 0, "density"] = 1
    print(f"Initial conditions created: {time.time()-start:.2f}s")

    for it in tqdm(range(iterations-1), desc="Algorithm position iteration"):
        
        # Calculate number of point with it=i if status=True
        previous = points.query("it == @it & status == True")
        
        # If there is none, set all points to True
        if previous.shape[0] == 0:
            print("No point with 'True' status!")
            previous = points.query("it == @it")

        # Points that are inside the FOV
        #previous = previous[previous.point.apply(cube.inside)]
        total    = previous.shape[0]
        volume   = 4/3 * numpy.pi * 15**3
        
        # For each it=i+1 calculate the number of points in sphere and probability
        current  = points.query("it == @it+1")
        current["density"] = current.sphere.apply(lambda sph: previous[previous.point.apply(sph.inside)].shape[0])
        current["density"] /= (total*volume)
        
        # Assign True or False to point
        points.loc[current.index,"status"]  = current.eval("choice <= density")
    print(f"Iterations elapsed: {time.time()-start:.2f}s")

    # Groupby ts, get last it, get True, calculate average position
    points["x"] = points.point.apply(lambda entry: entry.get_x())
    points["y"] = points.point.apply(lambda entry: entry.get_y())
    points["z"] = points.point.apply(lambda entry: entry.get_z())
    final_pos = points.groupby(["id"])[["x","y","z"]].mean().apply(
        lambda entry: 
        geometry.Point3(
                entry[0],
                entry[1],
                entry[2]
        ),
        axis=1
    )
    points["Final"] = points.id.apply(lambda x: final_pos[x])
    print(f"Final positions calculated: {time.time()-start:.2f}s")
        
    return points