from flask import Flask, render_template, request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import io, base64
import numpy as np
import networkx as nx
import random
import math

# ================= Flask =================
app = Flask(__name__)

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

# ================= Wi-Fi Routers =================
WIFI_ROUTERS = {
    "Router_DOAA": (12.18, 6.61),
    "Router_RnD": (41.0, 6.61),
}

WIFI_DETECTION_RANGE = 15.0

# ================= RSSI MODEL =================
RSSI_A = 36.5345
RSSI_B = 29.742

def distance_to_rssi(distance):
    """RSSI = -29.742 log10(d) + 36.5345"""
    distance = max(distance, 0.1)
    rssi = random.uniform(-20, -50)
    noise = random.uniform(-3, 3)
    return rssi + noise

def rssi_to_distance(rssi):
    """Inverse RSSI model"""
    return (10 ** ((RSSI_A - rssi) / RSSI_B))/100

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
        return "Move East"
    elif 45 <= angle < 135:
        return "Move North"
    elif -135 <= angle < -45:
        return "Move South"
    else:
        return "Move West"

def check_wifi_proximity(path_coords):
    closest = None
    min_dist = float('inf')
    idx = -1
    router_data = None

    for name, pos in WIFI_ROUTERS.items():
        for i, p in enumerate(path_coords):
            d = np.linalg.norm(np.array(p) - np.array(pos))
            if d < min_dist and d <= WIFI_DETECTION_RANGE:
                min_dist = d
                closest = p
                idx = i
                router_data = (name, pos)

    return closest, router_data, min_dist, idx

def simulate_wifi_localization(path_coords):
    curr, router, true_dist, idx = check_wifi_proximity(path_coords)

    if not router:
        return None, None, None, "Wi-Fi signal not detected"

    router_name, router_pos = router
    rssi = round(distance_to_rssi(true_dist), 2)
    est_dist = rssi_to_distance(rssi)

    direction = "Destination reached"
    if idx + 1 < len(path_coords):
        direction = get_direction(curr, path_coords[idx + 1])

    msg = (
        f"Wi-Fi Detected | Router: {router_name} | "
        f"RSSI: {rssi} dBm | Estimated Distance: {est_dist:.2f} m | {direction}"
    )

    return router_pos, est_dist, curr, msg

# ================= Visualization =================
def draw_map(G, path, start, goal, wifi_data=None):
    fig, ax = plt.subplots(figsize=(14, 9))

    for x, y, w, h in obstacles:
        ax.add_patch(Rectangle((x, y), w, h, color="gray", alpha=0.5))

    for name, (x, y) in LANDMARKS.items():
        ax.plot(x, y, 'bo')
        ax.text(x, y + 0.5, name, fontsize=7, ha='center')

    if path:
        coords = [G.nodes[n]['pos'] for n in path]
        xs, ys = zip(*coords)
        ax.plot(xs, ys, 'r-', linewidth=3)

    ax.plot(*start, 'go', markersize=12)
    ax.plot(*goal, 'rs', markersize=12)

    for name, (x, y) in WIFI_ROUTERS.items():
        ax.plot(x, y, 'm^', markersize=10)
        ax.text(x, y - 0.7, name, ha='center', color='purple')

    if wifi_data:
        router_pos, radius, curr_pos, _ = wifi_data
        ax.add_patch(Circle(router_pos, radius, color='yellow', alpha=0.3))
        ax.plot(curr_pos[0], curr_pos[1], 'y*', markersize=20)

    ax.set_xlim(0, ENV_WIDTH)
    ax.set_ylim(0, ENV_HEIGHT)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':')
    ax.set_title("Indoor Navigation with RSSI-Based Wi-Fi Localization")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ================= Route =================
@app.route("/", methods=["GET", "POST"])
def index():
    image = None
    wifi_message = None

    if request.method == "POST":
        start = LANDMARKS[request.form["start"]]
        goal = LANDMARKS[request.form["goal"]]

        prm = PRM(n_samples=300)
        G, path = prm.solve(start, goal)

        if path:
            coords = [G.nodes[n]['pos'] for n in path]
            wifi_data = simulate_wifi_localization(coords)
            image = draw_map(G, path, start, goal, wifi_data)
            wifi_message = wifi_data[3]

    return render_template(
        "index2.html",
        landmarks=LANDMARKS.keys(),
        image=image,
        wifi_message=wifi_message
    )

if __name__ == "__main__":
    app.run(debug=True, port=5001)
