from flask import Flask, render_template, request, jsonify
import heapq
from collections import deque

app = Flask(__name__)

# --- MÔ HÌNH DỮ LIỆU ĐỒ THỊ ---
class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.adj = {} 

    def add_node(self, id, label, type):
        self.nodes[id] = {'label': label, 'type': type}
        self.adj[id] = []

    def add_edge(self, u, v, w, capacity=0):
        # w: Trọng số (Distance/Cost)
        # c: Dung lượng (Capacity) cho Max Flow
        self.edges.append({'u': u, 'v': v, 'w': w, 'c': capacity})
        self.adj[u].append([v, w, capacity])
        # Giả định đồ thị vô hướng cho các bài toán tìm đường
        self.adj.setdefault(v, []).append([u, w, capacity])

    def get_dist(self, u, v):
        for item in self.adj.get(u, []):
            if item[0] == v: return item[1]
        return float('inf')

# --- THƯ VIỆN THUẬT TOÁN (ALGORITHMS LIBRARY) ---
class Algorithms:
    
    # 1. THUẬT TOÁN TÌM ĐƯỜNG NGẮN NHẤT (DIJKSTRA)
    # Dựa trên Slide Chương 8: Bài toán đường đi ngắn nhất
    @staticmethod
    def dijkstra(g, start, end):
        pq = [(0, start, [])] 
        visited = set()
        
        while pq:
            (cost, u, path) = heapq.heappop(pq)
            if u in visited: continue
            visited.add(u)
            path = path + [u]
            
            if u == end: return cost, path
            
            for v, w, c in g.adj[u]:
                if v not in visited:
                    heapq.heappush(pq, (cost + w, v, path))
        return 0, []

    # 2. THUẬT TOÁN LUỒNG CỰC ĐẠI (EDMONDS-KARP)
    # Dựa trên Slide Chương 8: Bài toán luồng cực đại
    @staticmethod
    def max_flow(g, source, sink):
        # Tạo đồ thị thặng dư
        capacity = {}
        for e in g.edges:
            capacity[(e['u'], e['v'])] = e['c']
            capacity[(e['v'], e['u'])] = e['c']
            
        flow = 0
        path_flows = [] 
        
        while True:
            parent = {node: None for node in g.nodes}
            queue = deque([source])
            path_found = False
            
            # BFS tìm đường tăng luồng
            while queue:
                u = queue.popleft()
                if u == sink:
                    path_found = True
                    break
                for v, w, c in g.adj[u]:
                    res_cap = capacity.get((u, v), 0)
                    if parent[v] is None and v != source and res_cap > 0:
                        parent[v] = u
                        queue.append(v)
                        
            if not path_found: break

            path_flow = float('inf')
            v = sink
            current_path = []
            while v != source:
                u = parent[v]
                current_path.append((u, v))
                path_flow = min(path_flow, capacity.get((u, v), 0))
                v = u
            current_path.reverse()
            
            flow += path_flow
            path_flows.append({'path': current_path, 'flow': path_flow})
            
            v = sink
            while v != source:
                u = parent[v]
                capacity[(u, v)] -= path_flow
                capacity[(v, u)] += path_flow
                v = u
                
        return flow, path_flows

    # 3. THUẬT TOÁN GIAO HÀNG TỐI ƯU (TSP 2-OPT)
    # Tối ưu hóa lộ trình đi qua nhiều điểm (Heuristic)
    @staticmethod
    def tsp_smart(g, start_node):
        nodes = list(g.nodes.keys())
        if not nodes: return 0, []
        
        # Bước 1: Nearest Neighbor
        path = [start_node]
        visited = {start_node}
        current = start_node
        
        while len(visited) < len(nodes):
            nearest = None
            min_dist = float('inf')
            for v, w, c in g.adj[current]:
                if v not in visited and w < min_dist:
                    min_dist = w
                    nearest = v
            
            if nearest is None:
                remain = [n for n in nodes if n not in visited]
                if not remain: break
                nearest = remain[0]
                min_dist = 100 
            
            visited.add(nearest)
            path.append(nearest)
            current = nearest
        
        path.append(start_node)
        
        # Bước 2: 2-Opt Optimization
        def get_route_dist(route):
            d = 0
            for i in range(len(route)-1):
                d += g.get_dist(route[i], route[i+1])
            return d

        best_dist = get_route_dist(path)
        improved = True
        for _ in range(50):
            improved = False
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path) - 1):
                    if j - i == 1: continue
                    new_path = path[:]
                    new_path[i:j] = path[i:j][::-1]
                    new_dist = get_route_dist(new_path)
                    if new_dist < best_dist:
                        path = new_path
                        best_dist = new_dist
                        improved = True
            if not improved: break
            
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        return best_dist, path_edges

# --- API ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    g = Graph()
    
    for n in data['nodes']:
        g.add_node(n['id'], n['label'], n['group'])
    for e in data['edges']:
        val = int(''.join(filter(str.isdigit, str(e['label']))))
        w = val if data['algo'] != 'MaxFlow' else 1
        c = val if data['algo'] == 'MaxFlow' else 0
        g.add_edge(e['from'], e['to'], w, c)

    algo = data['algo']
    src, sink = int(data['src']), int(data['sink'])
    res = {'status': 'ok', 'logs': [], 'visuals': []}

    try:
        if algo == 'Dijkstra':
            dist, path = Algorithms.dijkstra(g, src, sink)
            res['logs'].append(f"🏁 [Shortest Path] Đã tìm thấy lộ trình tối ưu.")
            res['logs'].append(f"📏 Tổng chi phí/quãng đường: {dist}")
            edges = [(path[i], path[i+1]) for i in range(len(path)-1)] if path else []
            res['visuals'] = [{'type': 'path', 'edges': edges, 'color': '#2ecc71'}]

        elif algo == 'MaxFlow':
            max_f, flows = Algorithms.max_flow(g, src, sink)
            res['logs'].append(f"🌊 [Network Flow] Phân tích luồng mạng lưới.")
            res['logs'].append(f"🚛 Khả năng vận chuyển cực đại: {max_f}")
            for i, flow in enumerate(flows):
                res['visuals'].append({
                    'type': 'flow', 
                    'edges': flow['path'], 
                    'val': flow['flow'],
                    'color': '#3498db'
                })

        elif algo == 'TSP':
            dist, edges = Algorithms.tsp_smart(g, src)
            res['logs'].append(f"📦 [Route Optimization] Đã tối ưu hóa lộ trình giao hàng.")
            res['logs'].append(f"🚚 Tổng quãng đường di chuyển: {dist}")
            res['visuals'] = [{'type': 'path', 'edges': edges, 'color': '#e74c3c'}]

    except Exception as e:
        res['status'] = 'error'
        res['logs'].append(f"System Error: {str(e)}")

    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=True)