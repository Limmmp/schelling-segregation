#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модель сегрегации Шеллинга — агентное моделирование на сетке.

Два типа агентов размещаются на квадратной сетке; часть клеток пуста.
Каждый агент характеризуется «терпимостью» T — максимальной долей соседей
другого типа, которую он готов принять. Если доля «чужих» среди соседей
превышает T, агент недоволен и переезжает в случайную свободную клетку.

Режимы работы:
  animate     — анимация перемещений агентов + график индекса сегрегации
                по поколениям (в реальном времени);
  experiment  — серия прогонов при разных значениях терпимости и построение
                зависимости итогового индекса сегрегации от терпимости.

Зависимости: numpy, matplotlib (кроссплатформенно: Windows, Linux, macOS).

Примеры запуска:
  python schelling.py animate --size 50 --density 0.9 --tolerance 0.4
  python schelling.py animate --gif demo.gif
  python schelling.py experiment --runs 5 --out segregation_vs_tolerance.png
"""

import argparse
import csv
import sys

import numpy as np

EMPTY, TYPE_A, TYPE_B = 0, 1, 2


class SchellingModel:
    """Модель Шеллинга на квадратной сетке size x size.

    tolerance — максимально допустимая для агента доля соседей другого
    типа (0 — не терпит ни одного «чужого», 1 — терпит любых соседей).
    """

    def __init__(self, size=50, density=0.9, ratio=0.5, tolerance=0.4, seed=None):
        self.size = size
        self.density = density
        self.ratio = ratio
        self.tolerance = tolerance
        self.rng = np.random.default_rng(seed)
        self.grid = self._init_grid()
        self.generation = 0

    def _init_grid(self):
        """Случайное начальное размещение агентов на сетке."""
        n_cells = self.size * self.size
        n_agents = int(n_cells * self.density)
        n_a = int(n_agents * self.ratio)
        n_b = n_agents - n_a
        cells = np.full(n_cells, EMPTY, dtype=np.int8)
        cells[:n_a] = TYPE_A
        cells[n_a:n_a + n_b] = TYPE_B
        self.rng.shuffle(cells)
        return cells.reshape(self.size, self.size)

    def _neighbour_counts(self):
        """Число соседей каждого типа для каждой клетки (окрестность Мура).

        Сетка не замкнута: у краевых клеток соседей меньше.
        Возвращает (same_a, same_b, occupied) — счётчики по 8 соседям.
        """
        a = (self.grid == TYPE_A).astype(np.int16)
        b = (self.grid == TYPE_B).astype(np.int16)

        def window_sum(m):
            p = np.pad(m, 1)
            return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
                    p[1:-1, :-2] + p[1:-1, 2:] +
                    p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])

        cnt_a = window_sum(a)
        cnt_b = window_sum(b)
        return cnt_a, cnt_b, cnt_a + cnt_b

    def unhappy_mask(self):
        """Маска недовольных агентов: доля «чужих» соседей превышает терпимость."""
        cnt_a, cnt_b, occupied = self._neighbour_counts()
        other = np.where(self.grid == TYPE_A, cnt_b,
                         np.where(self.grid == TYPE_B, cnt_a, 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            frac_other = np.where(occupied > 0, other / occupied, 0.0)
        return (self.grid != EMPTY) & (frac_other > self.tolerance)

    def segregation_index(self):
        """Индекс сегрегации: средняя доля соседей своего типа.

        Считается по агентам, имеющим хотя бы одного соседа.
        0.5 — полное перемешивание при равных долях, 1.0 — полная сегрегация.
        """
        cnt_a, cnt_b, occupied = self._neighbour_counts()
        same = np.where(self.grid == TYPE_A, cnt_a,
                        np.where(self.grid == TYPE_B, cnt_b, 0))
        mask = (self.grid != EMPTY) & (occupied > 0)
        if not mask.any():
            return 0.0
        return float(np.mean(same[mask] / occupied[mask]))

    def step(self):
        """Одно поколение: все недовольные агенты одновременно переезжают
        в случайные свободные клетки. Возвращает число переехавших."""
        unhappy = np.argwhere(self.unhappy_mask())
        if len(unhappy) == 0:
            return 0
        movers = self.grid[unhappy[:, 0], unhappy[:, 1]].copy()
        # освобождаем клетки недовольных; свободными становятся и старые
        # пустые клетки, и только что покинутые
        self.grid[unhappy[:, 0], unhappy[:, 1]] = EMPTY
        empty = np.argwhere(self.grid == EMPTY)
        dest_idx = self.rng.choice(len(empty), size=len(movers), replace=False)
        dest = empty[dest_idx]
        order = self.rng.permutation(len(movers))
        self.grid[dest[:, 0], dest[:, 1]] = movers[order]
        self.generation += 1
        return len(movers)

    def run(self, max_generations=200):
        """Прогон до сходимости (нет недовольных) или до max_generations.

        Возвращает список значений индекса сегрегации по поколениям.
        """
        history = [self.segregation_index()]
        for _ in range(max_generations):
            moved = self.step()
            history.append(self.segregation_index())
            if moved == 0:
                break
        return history


# ----------------------------- визуализация -----------------------------

def make_colormap():
    from matplotlib.colors import ListedColormap
    # пустая клетка — белая, тип A — синий, тип B — красный
    return ListedColormap(["#f5f5f5", "#1f77b4", "#d62728"])


def cmd_animate(args):
    import matplotlib
    if args.gif:
        matplotlib.use("Agg")  # для записи GIF окно не требуется
    else:
        # для интерактивного окна принудительно выбираем Tk-бэкенд;
        # без этого в собранном исполняемом файле matplotlib откатывается
        # на неинтерактивный Agg и окно не отображается
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
    import matplotlib.pyplot as plt
    from matplotlib import animation

    model = SchellingModel(size=args.size, density=args.density,
                           ratio=args.ratio, tolerance=args.tolerance,
                           seed=args.seed)
    history = [model.segregation_index()]

    fig, (ax_grid, ax_plot) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.suptitle("Модель сегрегации Шеллинга "
                 f"(терпимость T = {args.tolerance:.2f})")

    im = ax_grid.imshow(model.grid, cmap=make_colormap(), vmin=0, vmax=2,
                        interpolation="nearest")
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    title = ax_grid.set_title("Поколение 0")

    line, = ax_plot.plot(history, color="#2ca02c")
    ax_plot.set_xlabel("Поколение")
    ax_plot.set_ylabel("Индекс сегрегации")
    ax_plot.set_ylim(0.4, 1.02)
    ax_plot.set_xlim(0, 20)
    ax_plot.grid(True, alpha=0.3)

    state = {"done": False}

    def update(_frame):
        if not state["done"]:
            moved = model.step()
            history.append(model.segregation_index())
            if moved == 0 or model.generation >= args.steps:
                state["done"] = True
        im.set_data(model.grid)
        title.set_text(f"Поколение {model.generation}"
                       + ("  (равновесие)" if state["done"] else ""))
        line.set_data(range(len(history)), history)
        ax_plot.set_xlim(0, max(20, len(history)))
        return im, line, title

    frames = args.steps if args.gif else None
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=args.interval, blit=False,
                                   cache_frame_data=False)
    fig.tight_layout()

    if args.gif:
        anim.save(args.gif, writer=animation.PillowWriter(fps=8))
        print(f"Анимация сохранена: {args.gif}")
    else:
        plt.show()
    print(f"Итог: поколение {model.generation}, "
          f"индекс сегрегации {history[-1]:.3f}")


def cmd_experiment(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tolerances = np.round(np.arange(0.0, 1.0001, args.tstep), 4)
    means, stds = [], []
    rows = []
    for t in tolerances:
        finals = []
        for r in range(args.runs):
            model = SchellingModel(size=args.size, density=args.density,
                                   ratio=args.ratio, tolerance=float(t),
                                   seed=None if args.seed is None
                                   else args.seed + r)
            history = model.run(max_generations=args.steps)
            finals.append(history[-1])
        means.append(np.mean(finals))
        stds.append(np.std(finals))
        rows.append([t, means[-1], stds[-1]])
        print(f"T = {t:.2f}: индекс сегрегации = {means[-1]:.3f} "
              f"± {stds[-1]:.3f} ({args.runs} прогонов)")

    csv_path = args.out.rsplit(".", 1)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["терпимость", "индекс сегрегации (среднее)",
                    "стандартное отклонение"])
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(tolerances, means, yerr=stds, marker="o", capsize=3,
                color="#1f77b4", ecolor="#aaaaaa")
    ax.set_xlabel("Терпимость T (допустимая доля соседей другого типа)")
    ax.set_ylabel("Итоговый индекс сегрегации")
    ax.set_title("Зависимость сегрегации от терпимости агентов\n"
                 f"(сетка {args.size}x{args.size}, заполненность "
                 f"{args.density:.0%}, {args.runs} прогонов на точку)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"График сохранён: {args.out}")
    print(f"Данные сохранены: {csv_path}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Модель сегрегации Шеллинга (агентное моделирование).")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--size", type=int, default=50,
                        help="размер сетки (по умолчанию 50)")
        sp.add_argument("--density", type=float, default=0.9,
                        help="доля занятых клеток (по умолчанию 0.9)")
        sp.add_argument("--ratio", type=float, default=0.5,
                        help="доля агентов первого типа (по умолчанию 0.5)")
        sp.add_argument("--steps", type=int, default=200,
                        help="максимум поколений (по умолчанию 200)")
        sp.add_argument("--seed", type=int, default=None,
                        help="зерно генератора случайных чисел")

    pa = sub.add_parser("animate", help="анимация модели")
    common(pa)
    pa.add_argument("--tolerance", type=float, default=0.4,
                    help="терпимость агентов, 0..1 (по умолчанию 0.4)")
    pa.add_argument("--interval", type=int, default=150,
                    help="задержка между кадрами, мс (по умолчанию 150)")
    pa.add_argument("--gif", type=str, default=None,
                    help="сохранить анимацию в GIF вместо показа окна")

    pe = sub.add_parser("experiment",
                        help="зависимость сегрегации от терпимости")
    common(pe)
    pe.add_argument("--tstep", type=float, default=0.1,
                    help="шаг перебора терпимости (по умолчанию 0.1)")
    pe.add_argument("--runs", type=int, default=5,
                    help="число прогонов на точку (по умолчанию 5)")
    pe.add_argument("--out", type=str, default="segregation_vs_tolerance.png",
                    help="файл графика (по умолчанию "
                         "segregation_vs_tolerance.png)")
    return p


def settings_dialog():
    """Окно выбора параметров перед запуском модели.

    Показывается при запуске без аргументов (например, двойным щелчком по
    исполняемому файлу). Возвращает список аргументов командной строки или
    None, если пользователь закрыл окно, не запустив модель.
    """
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.title("Модель сегрегации Шеллинга")
    root.resizable(False, False)

    mode = tk.StringVar(value="animate")
    # ключ: (подпись, значение по умолчанию, только для режимов)
    fields = [
        ("size", "Размер сетки N", "50", ("animate", "experiment")),
        ("density", "Плотность заселения (0..1)", "0.9", ("animate", "experiment")),
        ("ratio", "Доля агентов 1-го типа (0..1)", "0.5", ("animate", "experiment")),
        ("tolerance", "Терпимость T (0..1)", "0.4", ("animate",)),
        ("steps", "Лимит поколений", "200", ("animate", "experiment")),
        ("seed", "Зерно (пусто — случайно)", "", ("animate", "experiment")),
        ("tstep", "Шаг терпимости", "0.1", ("experiment",)),
        ("runs", "Прогонов на точку", "5", ("experiment",)),
    ]
    entries, labels = {}, {}
    result = {"argv": None}

    tk.Label(root, text="Параметры моделирования",
             font=("Segoe UI", 11, "bold")).grid(
        row=0, column=0, columnspan=2, padx=12, pady=(12, 8))

    frm_mode = tk.Frame(root)
    frm_mode.grid(row=1, column=0, columnspan=2, pady=(0, 8))
    tk.Label(frm_mode, text="Режим:").pack(side="left", padx=(0, 6))
    tk.Radiobutton(frm_mode, text="Анимация", variable=mode,
                   value="animate").pack(side="left")
    tk.Radiobutton(frm_mode, text="Эксперимент", variable=mode,
                   value="experiment").pack(side="left")

    for i, (key, label, default, _modes) in enumerate(fields, start=2):
        lbl = tk.Label(root, text=label, anchor="w")
        lbl.grid(row=i, column=0, sticky="w", padx=(12, 6), pady=2)
        ent = tk.Entry(root, width=14)
        ent.insert(0, default)
        ent.grid(row=i, column=1, sticky="e", padx=(0, 12), pady=2)
        entries[key], labels[key] = ent, lbl

    def update_state(*_):
        # неактивные для выбранного режима поля показываем серым
        for key, _label, _default, modes in fields:
            active = mode.get() in modes
            entries[key].config(state="normal" if active else "disabled")
            labels[key].config(fg="black" if active else "gray")

    mode.trace_add("write", update_state)
    update_state()

    def on_run():
        m = mode.get()
        v = {k: entries[k].get().strip() for k in entries}
        try:
            int(v["size"]); int(v["steps"])
            float(v["density"]); float(v["ratio"])
            if v["seed"]:
                int(v["seed"])
            if m == "animate":
                float(v["tolerance"])
            else:
                float(v["tstep"]); int(v["runs"])
        except ValueError:
            messagebox.showerror("Ошибка ввода",
                                 "Проверьте правильность введённых значений.")
            return
        argv = [m, "--size", v["size"], "--density", v["density"],
                "--ratio", v["ratio"], "--steps", v["steps"]]
        if v["seed"]:
            argv += ["--seed", v["seed"]]
        if m == "animate":
            argv += ["--tolerance", v["tolerance"]]
        else:
            argv += ["--tstep", v["tstep"], "--runs", v["runs"]]
        result["argv"] = argv
        root.destroy()

    tk.Button(root, text="Запустить", width=16, command=on_run).grid(
        row=len(fields) + 2, column=0, columnspan=2, pady=(10, 12))

    root.eval("tk::PlaceWindow . center")
    root.mainloop()
    return result["argv"]


def main(argv=None):
    # корректный вывод кириллицы в консоли Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if argv is None:
        argv = sys.argv[1:]
    # запуск без аргументов (например, двойным щелчком по файлу) —
    # предлагаем выбрать параметры в диалоговом окне
    if not argv:
        try:
            argv = settings_dialog()
        except Exception:
            argv = ["animate"]  # если графическая среда недоступна
        if argv is None:
            return  # пользователь закрыл окно, не запустив модель
    args = build_parser().parse_args(argv)
    if args.command == "animate":
        cmd_animate(args)
    elif args.command == "experiment":
        cmd_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
