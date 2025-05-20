import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines

# ====== Model Functions ======

def phys_model(t, a, k):
    return a * np.sin(k * t)

class synth_model:
    def __init__(self, t):
        self.t = t
        self.state = np.zeros_like(t)  # 25 time steps
    def __call__(self, t):
        return np.interp(t, self.t, self.state)
    def update(self, new):
        self.state = new

def synth_model_deformed(t, synth_model, ghost_t, phys_a, phys_k, sigma=0.8):
    y = synth_model.state
    for gt in ghost_t:
        delta = 0.1 * (phys_model(gt, phys_a, phys_k) - synth_model(gt))
        y += delta * np.exp(-0.5 * ((t - gt) / sigma) ** 2)
    for st, xt in zip(sensors, x_sensors):
        delta = 0.1 * (xt - synth_model(st))
        y += delta * np.exp(-0.5 * ((t - st) / sigma) ** 2)
    synth_model.update(y)
    return y

drawn_arrows = []
   

# ====== Data Setup ======

t = np.linspace(0, 2*np.pi, 1000)
sensors = np.array([0, 1, 3, 5, 6])
x_sensors = np.sin(sensors)

steps = []
s_m = synth_model(t)
ghost_t = np.array([])

high_level_steps = [
    ("Initial State", (0.5,1), (0.0,0.0), False, None),
    ("Place Ghost Sensors", (0.5,1), (0.0,0.0), True, None),
    ("Epoch 1: Train Synthetic Model", (0.5,1), (0.0,0.0), False, "synth_arrows"),
    ("Epoch 1: Train Synthetic Model", (0.5,1), (0.0,0.0), False, 'synth'),
    ("Epoch 1: Train Physical Model", (0.5,1), (0.75,1), False, "phys_arrows"),
    ("Epoch 1: Train Physical Model", (0.5,1), (0.75,1), False, 'phys'),

    ("Place Ghost Sensors", (0.75,1), (0.75,1), True, None),
    ("Epoch 2: Train Synthetic Model", (0.75,1), (0.75,1), False, "synth_arrows"),
    ("Epoch 2: Train Synthetic Model", (0.75,1), (0.75,1), False, 'synth'),
    ("Epoch 2: Train Physical Model", (0.75,1), (0.9,1), False, "phys_arrows"),
    ("Epoch 2: Train Physical Model", (0.75,1), (0.9,1), False, 'phys'),

    ("Place Ghost Sensors", (0.9,1), (0.9,1), True, None),
    ("Epoch 3: Train Synthetic Model", (0.9,1), (0.9,1), False, "synth_arrows"),
    ("Epoch 3: Train Synthetic Model", (0.9,1), (0.9,1), False, 'synth'),
    ("Epoch 3: Train Physical Model", (0.9,1), (0.95,1), False, "phys_arrows"),
    ("Epoch 3: Train Physical Model", (0.9,1), (0.95,1), False, 'phys'),

    ("Place Ghost Sensors", (0.95,1), (0.95,1), True, None),
    ("Epoch 4: Train Synthetic Model", (0.95,1), (0.95,1), False, "synth_arrows"),
    ("Epoch 4: Train Synthetic Model", (0.95,1), (0.95,1), False, 'synth'),
    ("Epoch 4: Train Physical Model", (0.95,1), (1,1), False, "phys_arrows"),
    ("Epoch 4: Train Physical Model", (0.95,1), (1,1), False, 'phys'),
]
for title, phys_pk, next_pk, place_ghosts, mode in high_level_steps:
    if place_ghosts:
        ghost_t = np.random.uniform(0, 2*np.pi, 5)
        for i in range(25):
            steps.append(('ghosts', title, phys_pk, next_pk, ghost_t.copy(), None))
    elif mode == 'synth_arrows':
        for i in range(15):
            steps.append(('synth_arrows', title, phys_pk, next_pk, ghost_t.copy(), None))
    elif mode == 'phys_arrows':
        for i in range(15):
            steps.append(('phys_arrows', title, phys_pk, next_pk, ghost_t.copy(), None))

    elif mode == 'synth':
        for i in range(25):
            synth_model_deformed(t, s_m, ghost_t, *phys_pk)
            steps.append(('synth', title, phys_pk, next_pk, ghost_t.copy(), s_m.state.copy()))
    elif mode == 'phys':
        for i in range(25):
            interp_a = phys_pk[0] + (next_pk[0] - phys_pk[0]) * i / 24
            interp_k = phys_pk[1] + (next_pk[1] - phys_pk[1]) * i / 24
            steps.append(('phys', title, (interp_a, interp_k), next_pk, ghost_t.copy(), None))
    else:
        for i in range(25):
            steps.append(('static', title, phys_pk, next_pk, ghost_t.copy(), None))

# ====== Plotting Setup ======

fig, ax = plt.subplots(figsize=(8,5))
fig.tight_layout()
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1.2, 1.2)
ax.set_xticks([])
ax.set_yticks([])

# Static artists
sensor_pts, = ax.plot([], [], 'ro', markersize=8)
ghost_pts_synth,  = ax.plot([], [], 'ko', markersize=5)
ghost_pts_phys,   = ax.plot([], [], 'ko', markersize=5)
synth_line, = ax.plot([], [], 'g-', lw=2)
phys_line,  = ax.plot([], [], 'b-', lw=2)
title_text = ax.text(0.5,1.05,"", ha='center', va='bottom', fontsize=14)

# create proxy artists for a fixed legend
sensors_handle = mlines.Line2D([], [], color='r', marker='o',
                                linestyle='None', label='Sensors')
ghosts_handle  = mlines.Line2D([], [], color='k', marker='o',
                                linestyle='None', label='Ghost Sensors')
phys_handle    = mlines.Line2D([], [], color='b', linestyle='-',
                                label='Physical Model')
synth_handle   = mlines.Line2D([], [], color='g', linestyle='-',
                                label='Synthetic Model')

# place the legend once (will not be cleared by animation)
fig.legend(handles=[sensors_handle, ghosts_handle, phys_handle, synth_handle],
            loc='upper right', bbox_to_anchor=(0.95, 0.95))

# ====== Animation Update ======

def init():
    sensor_pts.set_data(sensors, x_sensors)
    s_m.update(np.zeros_like(t))
    return sensor_pts, ghost_pts_synth, ghost_pts_phys, synth_line, phys_line, title_text

def animate(i):
    # arrowprops produce FancyArrowPatch instances in ax.patches
    global drawn_arrows
    
    for arrow in drawn_arrows:
        arrow.remove()
    drawn_arrows.clear()

    step = steps[i]
    step_type, title, (a_phys, k_phys), _, ghost_t, synth_state = step
    if synth_state is not None:
        s_m.update(synth_state)
    ghost_pts_synth.set_data(ghost_t, s_m(ghost_t))
    ghost_pts_phys.set_data(ghost_t, phys_model(ghost_t, a_phys, k_phys))

    
    title_text.set_text(title)
    title_text.set_position((3, 1.0))

    y_phys = phys_model(t, a_phys, k_phys)
    y_synth = s_m(t)

    y_phys_ghost = phys_model(ghost_t, a_phys, k_phys)
    y_synth_ghost = s_m(ghost_t)

    y_phys_sensors = phys_model(sensors, a_phys, k_phys)
    y_synth_sensors = s_m(sensors)

    phys_line.set_data(t, y_phys)
    synth_line.set_data(t, y_synth)

    

    if step_type == 'ghosts':
        pass
    elif step_type == 'synth_arrows' or step_type == 'synth':
        
        for xi, ys, ye in zip(ghost_t, y_synth_ghost, y_phys_ghost):
            arrow = ax.annotate('', xy=(xi, ye), xytext=(xi, ys),
            arrowprops=dict(arrowstyle='-|>', color='k', shrinkA=0.0, shrinkB=0.0, linewidth=2),zorder=5)
            drawn_arrows.append(arrow)

        for xi, ys, ye in zip(sensors, y_synth_sensors, x_sensors):
            arrow = ax.annotate('', xy=(xi, ye), xytext=(xi, ys),
                arrowprops=dict(arrowstyle='-|>', color='red', shrinkA=0.0, shrinkB=0.0, linewidth=2),zorder=5)
            drawn_arrows.append(arrow)

    elif step_type == 'phys_arrows' or step_type == 'phys':
        for xi, ys, ye in zip(ghost_t, y_phys_ghost, y_synth_ghost):
            arrow = ax.annotate('', xy=(xi, ye), xytext=(xi, ys),
                arrowprops=dict(arrowstyle='-|>', color='k', shrinkA=0.0, shrinkB=0.0, linewidth=2),zorder=5)
            drawn_arrows.append(arrow)

        for xi, ys, ye in zip(sensors, y_phys_sensors, x_sensors):
            arrow = ax.annotate('', xy=(xi, ye), xytext=(xi, ys),
                arrowprops=dict(arrowstyle='-|>', color='red', shrinkA=0.0, shrinkB=0.0, linewidth=2),zorder=5)
            drawn_arrows.append(arrow)


    return sensor_pts, ghost_pts_synth, ghost_pts_phys, synth_line, phys_line, title_text, *drawn_arrows

# ====== Animate and Save ======

ani = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(steps), interval=100, blit=True)

ani.save('training_algorithm.gif', writer='pillow', fps=10, dpi=200)
print("Animation saved!")

