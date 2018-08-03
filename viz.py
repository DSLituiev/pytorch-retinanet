import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_bboxes(anc, clsf=None, ax=None,
                encoding='coord',
                edgecolor='r',
                facecolor='none',
                alpha=1.0,
                          ):
    """
    encoding in ('coord', 'size')
    """
    assert encoding in ('coord', 'size')
    coord_enc = encoding.startswith('coord')
    

    if ax is None:
        #fig, ax = plt.subplots(1)
        ax = plt.gca()
    if clsf is not None:
        alphas = 1 - clsf
    else:
        alphas = [alpha] * len(anc)

    for bbox, aa in zip(anc, alphas):
        x0, y0, x1,y1 = bbox#[0]
        #print([int(x) for x in  [x0, y0, x1,y1]], aa)
        #     print(x1-x0,y1-y0)
        if coord_enc:
            w = x1-x0
            h = y1-y0
        else:
            w = x1
            h = y1
        rect = patches.Rectangle((x0, y0), w, h, linewidth=2,
                                 edgecolor=edgecolor,
                                 facecolor=facecolor,
                                 alpha = aa**2)
        ax.add_patch(rect)
    
    old_xlims = ax.get_xlim()
    if coord_enc:
        xmax = float(anc[:,2].max())
        ymax = float(anc[:,3].max())
    else:
        xmax = float((anc[:,2] + anc[:,0]).max())
        ymax = float((anc[:,3] + anc[:,1]).max())

    print(float(anc[:,0].min()), xmax)
    curr_xlims = 1.05*np.r_[float(anc[:,0].min()), xmax]
    new_xlims = [min(curr_xlims[0], old_xlims[0]),
                 max(curr_xlims[1], old_xlims[1])]
    
    old_ylims = sorted(ax.get_ylim())
    curr_ylims = 1.05*np.r_[float(anc[:,1].min()), ymax]
    new_ylims = [min(curr_ylims[0], old_ylims[0]), 
                 max(curr_ylims[1], old_ylims[1])]
    
    ax.set_xlim(new_xlims)
    ax.set_ylim(new_ylims[::-1])

