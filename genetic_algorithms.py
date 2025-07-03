import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GAApp:
    def __init__(self, root):
        self.root = root
        root.title("GA Vertex Cover")

        # State
        self.graph = None
        self.edges = []
        self.population = []
        self.best_scores = []
        self.mean_scores = []
        self.generation = 0

        self.params = {
            "num_vertices":     tk.IntVar(value=10),
            "num_edges":        tk.IntVar(value=15),
            "pop_size":         tk.IntVar(value=20),
            "generations":      tk.IntVar(value=30),
            "mutation_rate":    tk.DoubleVar(value=0.1),
            "crossover_rate":   tk.DoubleVar(value=0.8),
            "selection_method": tk.StringVar(value="tournament")
        }

        ctrl = ttk.Frame(root)
        ctrl.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        ttk.Label(ctrl, text="Вершины:").grid(row=0, column=0, sticky="w")
        ttk.Entry(ctrl, textvariable=self.params["num_vertices"], width=5).grid(row=0, column=1)
        ttk.Label(ctrl, text="Рёбра:").grid(row=0, column=2, sticky="w", padx=(10,0))
        ttk.Entry(ctrl, textvariable=self.params["num_edges"], width=5).grid(row=0, column=3)

        ttk.Label(ctrl, text="Популяция:").grid(row=1, column=0, sticky="w")
        ttk.Entry(ctrl, textvariable=self.params["pop_size"], width=5).grid(row=1, column=1)
        ttk.Label(ctrl, text="Поколения:").grid(row=1, column=2, sticky="w", padx=(10,0))
        ttk.Entry(ctrl, textvariable=self.params["generations"], width=5).grid(row=1, column=3)

        ttk.Label(ctrl, text="Мутация:").grid(row=2, column=0, sticky="w")
        ttk.Entry(ctrl, textvariable=self.params["mutation_rate"], width=5).grid(row=2, column=1)
        ttk.Label(ctrl, text="Кроссовер:").grid(row=2, column=2, sticky="w", padx=(10,0))
        ttk.Entry(ctrl, textvariable=self.params["crossover_rate"], width=5).grid(row=2, column=3)

        ttk.Label(ctrl, text="Селекция:").grid(row=3, column=0, sticky="w")
        ttk.Combobox(ctrl,
                     textvariable=self.params["selection_method"],
                     values=["tournament","roulette"],
                     width=10).grid(row=3, column=1)

        ttk.Label(ctrl, text="Рёбра вручную (0-1,0-2):").grid(row=4, column=0, columnspan=4, sticky="w", pady=(10,0))
        self.edges_input = tk.Text(ctrl, width=20, height=4)
        self.edges_input.grid(row=5, column=0, columnspan=4)

        self.use_manual = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Использовать ручные рёбра", variable=self.use_manual).grid(
            row=6, column=0, columnspan=2, sticky="w")

        ttk.Button(ctrl, text="Загрузить файл", command=self.load_file).grid(row=7, column=0, columnspan=2, pady=5)
        ttk.Button(ctrl, text="Инициализировать", command=self.initialize).grid(row=7, column=2, columnspan=2, pady=5)

        ttk.Button(ctrl, text="Следующий шаг", command=self.next_step).grid(row=8, column=0, pady=5)
        ttk.Label(ctrl, text="Пропустить").grid(row=8, column=1)
        self.skip_entry = ttk.Entry(ctrl, width=5)
        self.skip_entry.insert(0, "5")
        self.skip_entry.grid(row=8, column=2)
        ttk.Button(ctrl, text="Пропустить N", command=self.skip_steps).grid(row=8, column=3)

        ttk.Label(ctrl, text="Список рёбер:").grid(row=9, column=0, columnspan=4, sticky="w", pady=(10,0))
        self.edges_display = tk.Text(ctrl, width=20, height=6, state="disabled")
        self.edges_display.grid(row=10, column=0, columnspan=4)

        # Graph canvas
        fig1 = plt.Figure(figsize=(4,4))
        self.ax_graph = fig1.add_subplot(111)
        self.canvas_graph = FigureCanvasTkAgg(fig1, master=root)
        self.canvas_graph.get_tk_widget().grid(row=0, column=1, rowspan=2, padx=10, pady=10)

        fig2 = plt.Figure(figsize=(4,3))
        self.ax_fit = fig2.add_subplot(111)
        self.canvas_fit = FigureCanvasTkAgg(fig2, master=root)
        self.canvas_fit.get_tk_widget().grid(row=2, column=1, padx=10, pady=10)

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files","*.txt"),("All files","*.*")])
        if not path:
            return
        with open(path, "r") as f:
            content = f.read()
        self.edges_input.delete("1.0", "end")
        self.edges_input.insert("1.0", content)
        manual = self.parse_edges(content)
        if manual:
            max_v = max(max(u, v) for u, v in manual)
            self.params["num_vertices"].set(max_v + 1)
            self.use_manual.set(True)
        messagebox.showinfo("Загрузка", f"Загружено {len(manual)} ребер")
        self.initialize()

    def parse_edges(self, text):
        edges = []
        for tok in text.replace("\n", ",").split(","):
            tok = tok.strip()
            if "-" in tok:
                u, v = tok.split("-", 1)
                try:
                    edges.append((int(u), int(v)))
                except:
                    pass
        return edges

    def initialize(self):
        n = self.params["num_vertices"].get()
        manual = self.parse_edges(self.edges_input.get("1.0", "end"))
        if self.use_manual.get() and manual:
            G = nx.Graph()
            G.add_nodes_from(range(n))
            for u, v in manual:
                if 0 <= u < n and 0 <= v < n:
                    G.add_edge(u, v)
        else:
            m = self.params["num_edges"].get()
            G = nx.Graph()
            G.add_nodes_from(range(n))
            while G.number_of_edges() < m:
                u, v = random.sample(range(n), 2)
                G.add_edge(u, v)
        self.graph = G
        self.edges = list(G.edges())
        self.update_edges_display()

        self.population = [self.generate_individual(n) for _ in range(self.params["pop_size"].get())]
        self.best_scores.clear()
        self.mean_scores.clear()
        self.generation = 0

        init_scores = [self.fitness(ind, self.edges) for ind in self.population]
        self.best_scores.append(max(init_scores))
        self.mean_scores.append(np.mean(init_scores))

        self.ax_graph.clear()
        self.ax_fit.clear()
        self.canvas_graph.draw()
        self.canvas_fit.draw()

        self.update_visuals()

    def generate_individual(self, n):
        return [random.choice([0, 1]) for _ in range(n)]

    def fitness(self, ind, eds):
        uncovered = sum(1 for u, v in eds if not (ind[u] or ind[v]))
        return - (sum(ind) + 10 * uncovered)

    def tournament_selection(self, pop, scores):
        i1, i2 = random.sample(range(len(pop)), 2)
        return pop[i1] if scores[i1] > scores[i2] else pop[i2]

    def roulette_selection(self, pop, scores):
        min_s = min(scores)
        shifted = [s - min_s + 1e-6 for s in scores]
        total = sum(shifted)
        probs = [s / total for s in shifted]
        return pop[np.random.choice(len(pop), p=probs)]

    def step_ga(self):
        pop = self.population
        eds = self.edges
        m_rate = self.params["mutation_rate"].get()
        c_rate = self.params["crossover_rate"].get()
        method = self.params["selection_method"].get()

        scores = [self.fitness(ind, eds) for ind in pop]
        new_pop = []
        for _ in pop:
            if method == "roulette":
                p1 = self.roulette_selection(pop, scores)
                p2 = self.roulette_selection(pop, scores)
            else:
                p1 = self.tournament_selection(pop, scores)
                p2 = self.tournament_selection(pop, scores)
            if random.random() < c_rate:
                child = self.crossover(p1, p2)
            else:
                child = p1.copy()
            new_pop.append(self.mutate(child, m_rate))

        self.population = new_pop
        self.best_scores.append(max(scores))
        self.mean_scores.append(np.mean(scores))
        self.generation += 1

    def next_step(self):
        if self.generation < self.params["generations"].get():
            self.step_ga()
            self.update_visuals()
        else:
            messagebox.showinfo("Info", "Максимум поколений достигнут")

    def skip_steps(self):
        try:
            k = int(self.skip_entry.get())
        except:
            return
        for _ in range(k):
            if self.generation < self.params["generations"].get():
                self.step_ga()
        self.update_visuals()

    def crossover(self, p1, p2):
        pt = random.randint(1, len(p1) - 1)
        return p1[:pt] + p2[pt:]

    def mutate(self, ind, rate):
        return [(1 - g if random.random() < rate else g) for g in ind]

    def update_edges_display(self):
        self.edges_display.config(state="normal")
        self.edges_display.delete("1.0", "end")
        for u, v in self.edges:
            self.edges_display.insert("end", f"{u} - {v}\n")
        self.edges_display.config(state="disabled")

    def update_visuals(self):
        cover = max(self.population, key=lambda ind: self.fitness(ind, self.edges))
        self.ax_graph.clear()
        pos = nx.spring_layout(self.graph, seed=42)
        colors = ['lightgreen' if cover[n] else 'lightgrey' for n in self.graph.nodes]
        nx.draw_networkx(self.graph, pos, ax=self.ax_graph, node_color=colors, with_labels=True)
        self.ax_graph.set_title(f"Поколение {self.generation}")
        self.ax_graph.axis("off")
        self.canvas_graph.draw()

        self.ax_fit.clear()
        self.ax_fit.plot(self.best_scores, label="Лучший", color="royalblue")
        self.ax_fit.plot(self.mean_scores, label="Средний", color="darkorange")
        self.ax_fit.set_xlabel("Поколение")
        self.ax_fit.set_ylabel("Fitness")
        self.ax_fit.legend()
        self.ax_fit.grid(True)

        gens = list(range(len(self.best_scores)))
        self.ax_fit.set_xticks(gens)
        self.ax_fit.set_xticklabels(gens, rotation=45, ha='right')
        self.ax_fit.relim()
        self.ax_fit.autoscale_view()

        self.canvas_fit.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = GAApp(root)
    root.mainloop()
