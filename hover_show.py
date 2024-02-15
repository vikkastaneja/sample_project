# https://learndataanalysis.org/source-code-display-info-when-you-hover-to-a-data-point-in-matplotlib/#google_vignette
import matplotlib.pyplot as plt
import numpy as np

# Step 1. Create a scatter chart
x = np.random.rand(20)
y = np.random.rand(20)
colors = np.random.randint(1, 5, size=len(x))
norm = plt.Normalize(1, 4)
cmap = plt.cm.PiYG

fig, ax = plt.subplots()
scatter = plt.scatter(
    x=x,
    y=y,
    c=colors,
    s=100,
    cmap=cmap,
    norm=norm
)

# Step 2. Create Annotation Object
annotation = ax.annotate(
    text='',
    xy=(0, 0),
    xytext=(15, 15), # distance from x, y
    textcoords='offset points',
    bbox={'boxstyle': 'round', 'fc': 'w'},
    arrowprops={'arrowstyle': '->'}
)
annotation.set_visible(False)


# Step 3. Implement the hover event to display annotations
def motion_hover(event):
    annotation_visbility = annotation.get_visible()
    if event.inaxes == ax:
        is_contained, annotation_index = scatter.contains(event)
        if is_contained:
            data_point_location = scatter.get_offsets()[annotation_index['ind'][0]]
            annotation.xy = data_point_location

            text_label = '({0:.2f}, {0:.2f})'.format(data_point_location[0], data_point_location[1])
            annotation.set_text(text_label)

            annotation.get_bbox_patch().set_facecolor(cmap(norm(colors[annotation_index['ind'][0]])))
            annotation.set_alpha(0.4)

            annotation.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if annotation_visbility:
                annotation.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', motion_hover)

plt.show()