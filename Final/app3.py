from flask import Flask, render_template, request, jsonify, session
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io, base64
import numpy as np
import networkx as nx
import random
import math

app = Flask(__name__)
app.secret_key = "indoor_nav_secret"

# ================= Environment =================
ENV_WIDTH = 46.0
ENV_HEIGHT = 26.68

obstacles = [
    (0.0, 0.0, 25.29, 6.61),
    (39.6, 0.0, 46.0 - 39.6, 6.61),
    (13.1, 8.81, 42.1 - 13.1, 21.81 - 8.81),
    (0.0, 8.81, 8.31, 26.68 - 8.81),
    (10.51, 24.68, 46.0 - 10.51, 26.68 - 24.68),
    (10.65, 11.88, 13.1 - 10.65, 21.81 - 11.88),
]

LANDMARKS = {
    "Main Gate": (35.69, 0),
    "R&D Office": (46, 6.8),
    "Finance Office": (46, 21.06),
    "Stores & Purchase": (34.59, 24.68),
    "Conference Room": (26.18, 21.81),
    "Washroom": (15.77, 24.68),
    "Electric Panel": (13.1, 21.81),
    "Staff Cafeteria": (9.31, 26.68),
    "DOSA Office": (8.31, 15.99),
    "Exit / Enter": (0, 7.81),
    "DOAA Office": (17.31, 6.61),
    "Office Automation Cell": (26.47, 8.81),
    "Elevator": (30.28, 5.69),
}

# ================= PRM =================
class PRM:
    def __init__(self, n_samples=500, k_nearest=15):
        self.n_samples = n_samples
        self.k = k_nearest

    def point_in_obstacle(self, p):
        x, y = p
        for ox, oy, w, h in obstacles:
            if ox < x < ox + w and oy < y < oy + h:
                return True
        return False

    def collision_free(self, p1, p2, checks=50):
        for t in np.linspace(0, 1, checks):
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if self.point_in_obstacle((x, y)):
                return False
        return True

    def build(self):
        samples = []
        while len(samples) < self.n_samples:
            p = (random.uniform(0, ENV_WIDTH),
                 random.uniform(0, ENV_HEIGHT))
            if not self.point_in_obstacle(p):
                samples.append(p)

        G = nx.Graph()
        for i, p in enumerate(samples):
            G.add_node(i, pos=p)

        for i in G.nodes:
            p1 = G.nodes[i]['pos']
            dists = []
            for j in G.nodes:
                if i != j:
                    p2 = G.nodes[j]['pos']
                    dists.append((j, np.linalg.norm(np.array(p1) - np.array(p2))))
            dists.sort(key=lambda x: x[1])

            for j, _ in dists[:self.k]:
                p2 = G.nodes[j]['pos']
                if self.collision_free(p1, p2):
                    G.add_edge(i, j, weight=np.linalg.norm(np.array(p1) - np.array(p2)))

        return G

    def solve(self, start, goal):
        G = self.build()
        G.add_node("start", pos=start)
        G.add_node("goal", pos=goal)

        for node in list(G.nodes):
            if node not in ["start", "goal"]:
                p = G.nodes[node]['pos']
                if self.collision_free(start, p):
                    G.add_edge("start", node, weight=np.linalg.norm(np.array(start) - np.array(p)))
                if self.collision_free(goal, p):
                    G.add_edge("goal", node, weight=np.linalg.norm(np.array(goal) - np.array(p)))

        try:
            path = nx.shortest_path(G, "start", "goal", weight="weight")
            return G, path
        except nx.NetworkXNoPath:
            return G, None

# ================= Helpers =================
def get_direction(curr, nxt):
    dx, dy = nxt[0] - curr[0], nxt[1] - curr[1]
    angle = math.degrees(math.atan2(dy, dx))
    if -45 <= angle < 45:
        return "Move East →"
    elif 45 <= angle < 135:
        return "Move North ↑"
    elif -135 <= angle < -45:
        return "Move South ↓"
    else:
        return "Move West ←"

def compute_turn_by_turn(G, path):
    """Generate human-readable turn-by-turn directions from path nodes."""
    if not path or len(path) < 2:
        return ["You are already at your destination."]

    coords = [G.nodes[n]['pos'] for n in path]
    directions = []
    step = 1

    i = 0
    while i < len(coords) - 1:
        direction = get_direction(coords[i], coords[i + 1])
        # Merge consecutive same-direction steps
        distance = 0
        while i < len(coords) - 1 and get_direction(coords[i], coords[i + 1]) == direction:
            distance += np.linalg.norm(np.array(coords[i + 1]) - np.array(coords[i]))
            i += 1
        directions.append(f"Step {step}: {direction} for ~{distance:.1f} m")
        step += 1

    directions.append(f"Step {step}: 🏁 You have arrived at your destination.")
    return directions

# ================= Visualization =================
def draw_map(G, path, start, goal, lost_landmark=None, is_reroute=False):
    fig, ax = plt.subplots(figsize=(14, 9))

    # Draw obstacles
    for x, y, w, h in obstacles:
        ax.add_patch(Rectangle((x, y), w, h, color="gray", alpha=0.5, label="_nolegend_"))

    # Draw landmarks
    for name, (x, y) in LANDMARKS.items():
        ax.plot(x, y, 'bo', markersize=6, zorder=5)
        ax.text(x, y + 0.4, name, fontsize=6.5, ha='center', color='navy', zorder=6)

    # Draw path
    if path:
        coords = [G.nodes[n]['pos'] for n in path]
        xs, ys = zip(*coords)
        color = 'darkorange' if is_reroute else 'red'
        label = 'Re-routed Path' if is_reroute else 'Planned Path'
        ax.plot(xs, ys, color=color, linewidth=3, label=label, zorder=4)

    # Start marker
    ax.plot(*start, 'go', markersize=14, zorder=7, label='Start')
    ax.text(start[0], start[1] - 1.0, 'START', fontsize=8, ha='center',
            color='green', fontweight='bold')

    # Goal marker
    ax.plot(*goal, 'r*', markersize=16, zorder=7, label='Goal')
    ax.text(goal[0], goal[1] - 1.0, 'GOAL', fontsize=8, ha='center',
            color='red', fontweight='bold')

    # Lost landmark marker
    if lost_landmark:
        ax.plot(*lost_landmark, 'y^', markersize=14, zorder=8, label="You are here (Lost)")
        ax.text(lost_landmark[0], lost_landmark[1] + 0.6, "📍 YOU ARE HERE",
                fontsize=8, ha='center', color='darkorange', fontweight='bold')

    ax.set_xlim(0, ENV_WIDTH)
    ax.set_ylim(0, ENV_HEIGHT)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', fontsize=8)
    title = " Re-routed Path (I'm Lost)" if is_reroute else "🗺️ Indoor Navigation — Planned Route"
    ax.set_title(title, fontsize=13, fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ================= Routes =================
@app.route("/", methods=["GET", "POST"])
def index():
    image = None
    directions = []
    error = None

    if request.method == "POST":
        start_name = request.form.get("start")
        goal_name = request.form.get("goal")

        if start_name == goal_name:
            error = "Start and destination cannot be the same location."
        else:
            start = LANDMARKS[start_name]
            goal = LANDMARKS[goal_name]

            # Save goal in session for "I'm Lost" re-routing
            session["goal_name"] = goal_name

            prm = PRM(n_samples=300)
            G, path = prm.solve(start, goal)

            if path:
                directions = compute_turn_by_turn(G, path)
                image = draw_map(G, path, start, goal)
            else:
                error = "No valid path found. Please try different locations."

    return render_template(
        "index2.html",
        landmarks=list(LANDMARKS.keys()),
        image=image,
        directions=directions,
        error=error,
        goal_name=session.get("goal_name", "")
    )


@app.route("/reroute", methods=["POST"])
def reroute():
    """Handle the 'I'm Lost' re-routing request."""
    lost_landmark_name = request.form.get("lost_landmark")
    goal_name = request.form.get("goal_name") or session.get("goal_name", "")

    if not lost_landmark_name or not goal_name:
        return jsonify({"error": "Please select both your current visible landmark and your destination."}), 400

    if lost_landmark_name == goal_name:
        return jsonify({"error": "You are already at your destination!"}), 400

    lost_pos = LANDMARKS[lost_landmark_name]
    goal_pos = LANDMARKS[goal_name]

    prm = PRM(n_samples=300)
    G, path = prm.solve(lost_pos, goal_pos)

    if not path:
        return jsonify({"error": "Could not find a path from your current location. Try a different landmark."}), 400

    directions = compute_turn_by_turn(G, path)
    image = draw_map(G, path, lost_pos, goal_pos,
                     lost_landmark=lost_pos, is_reroute=True)

    return jsonify({
        "image": image,
        "directions": directions,
        "message": f"Re-routed from {lost_landmark_name} to {goal_name}."
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)