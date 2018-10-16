import matplotlib.pyplot as plt

def make_barplots(results, holdout, names, title):
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    bp_dict = plt.boxplot(results)
    ax.set_xticklabels(names)

    for idx, line in enumerate(bp_dict['boxes']):
        x, y = line.get_xydata()[0] # bottom of left line
        plt.text(x+.25,y-0.005, 'H:%.2f' % holdout[idx],
            horizontalalignment='center', # centered
            verticalalignment='top')

    plt.show()    