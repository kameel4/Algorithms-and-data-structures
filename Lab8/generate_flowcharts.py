from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("flowcharts")
BG = "black"
FILL = "white"
LINE = "white"
TEXT = "black"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = font(30)
FONT_SMALL = font(24)


@dataclass
class Node:
    name: str
    kind: str
    text: str
    x: int
    y: int
    w: int = 460
    h: int = 118

    @property
    def left(self) -> int:
        return self.x - self.w // 2

    @property
    def right(self) -> int:
        return self.x + self.w // 2

    @property
    def top(self) -> int:
        return self.y - self.h // 2

    @property
    def bottom(self) -> int:
        return self.y + self.h // 2


class Flowchart:
    def __init__(self, height: int, width: int = 1100) -> None:
        self.image = Image.new("RGB", (width, height), BG)
        self.draw = ImageDraw.Draw(self.image)
        self.width = width
        self.height = height
        self.nodes: dict[str, Node] = {}

    def node(
        self,
        name: str,
        kind: str,
        text: str,
        x: int,
        y: int,
        w: int = 460,
        h: int = 118,
    ) -> Node:
        node = Node(name=name, kind=kind, text=text, x=x, y=y, w=w, h=h)
        self.nodes[name] = node
        self._draw_node(node)
        return node

    def _draw_node(self, node: Node) -> None:
        if node.kind == "terminal":
            self.draw.ellipse([node.left, node.top, node.right, node.bottom], fill=FILL, outline=LINE, width=3)
        elif node.kind == "decision":
            points = [(node.x, node.top), (node.right, node.y), (node.x, node.bottom), (node.left, node.y)]
            self.draw.polygon(points, fill=FILL, outline=LINE)
            self.draw.line(points + [points[0]], fill=LINE, width=3)
        elif node.kind == "io":
            skew = 48
            points = [
                (node.left + skew, node.top),
                (node.right, node.top),
                (node.right - skew, node.bottom),
                (node.left, node.bottom),
            ]
            self.draw.polygon(points, fill=FILL, outline=LINE)
            self.draw.line(points + [points[0]], fill=LINE, width=3)
        elif node.kind == "subprocess":
            self.draw.rectangle([node.left, node.top, node.right, node.bottom], fill=FILL, outline=LINE, width=3)
            self.draw.line([(node.left + 34, node.top), (node.left + 34, node.bottom)], fill=TEXT, width=3)
            self.draw.line([(node.right - 34, node.top), (node.right - 34, node.bottom)], fill=TEXT, width=3)
        else:
            self.draw.rectangle([node.left, node.top, node.right, node.bottom], fill=FILL, outline=LINE, width=3)
        self._text(node.text, node.x, node.y, node.w - 70, node.h - 20)

    def _text(self, text: str, cx: int, cy: int, max_w: int, max_h: int) -> None:
        chosen = FONT
        lines = self._wrap(text, chosen, max_w)
        line_h = self._line_height(chosen)
        if len(lines) * line_h > max_h:
            chosen = FONT_SMALL
            lines = self._wrap(text, chosen, max_w)
            line_h = self._line_height(chosen)
        total_h = len(lines) * line_h
        y = cy - total_h // 2
        for line in lines:
            bbox = self.draw.textbbox((0, 0), line, font=chosen)
            w = bbox[2] - bbox[0]
            self.draw.text((cx - w // 2, y), line, font=chosen, fill=TEXT)
            y += line_h

    def _wrap(self, text: str, used_font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
        result: list[str] = []
        for paragraph in text.split("\n"):
            words = paragraph.split()
            current = ""
            for word in words:
                trial = word if not current else f"{current} {word}"
                width = self.draw.textbbox((0, 0), trial, font=used_font)[2]
                if width <= max_w:
                    current = trial
                else:
                    if current:
                        result.append(current)
                    current = word
            if current:
                result.append(current)
        return result or [""]

    @staticmethod
    def _line_height(used_font: ImageFont.FreeTypeFont) -> int:
        bbox = used_font.getbbox("АБВabc123")
        return bbox[3] - bbox[1] + 10

    def arrow(self, start: str, end: str, label: str = "") -> None:
        a = self.nodes[start]
        b = self.nodes[end]
        self.polyline([(a.x, a.bottom), (b.x, b.top)], label)

    def polyline(self, points: list[tuple[int, int]], label: str = "") -> None:
        self.draw.line(points, fill=LINE, width=4)
        self._arrow_head(points[-2], points[-1])
        if label:
            mid = points[len(points) // 2]
            bbox = self.draw.textbbox((0, 0), label, font=FONT_SMALL)
            label_w = bbox[2] - bbox[0]
            x = mid[0] + 10
            if x + label_w > self.width - 10:
                x = mid[0] - label_w - 10
            x = max(10, x)
            y = max(10, mid[1] - 28)
            self.draw.text((x, y), label, font=FONT_SMALL, fill=LINE)

    def _arrow_head(self, p1: tuple[int, int], p2: tuple[int, int]) -> None:
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        size = 18
        left = (
            p2[0] - size * math.cos(angle - math.pi / 6),
            p2[1] - size * math.sin(angle - math.pi / 6),
        )
        right = (
            p2[0] - size * math.cos(angle + math.pi / 6),
            p2[1] - size * math.sin(angle + math.pi / 6),
        )
        self.draw.polygon([p2, left, right], fill=LINE)

    def save(self, name: str) -> None:
        OUT_DIR.mkdir(exist_ok=True)
        self.image.save(OUT_DIR / name)


def draw_main() -> None:
    c = Flowchart(2050)
    x = 550
    c.node("start", "terminal", "начало", x, 70, 300, 92)
    c.node("ui", "process", "открытие графического интерфейса", x, 205)
    c.node("task", "io", "выбор задачи и ввод параметров", x, 355)
    c.node("grid?", "decision", "выбрана задача сетки?", x, 540, 430, 150)
    c.node("grid", "subprocess", "запуск алгоритма поиска пути", 315, 760, 430, 130)
    c.node("puzzle", "subprocess", "запуск алгоритма пятнашек", 785, 760, 430, 130)
    c.node("metrics", "process", "измерение времени, памяти и состояний", x, 990)
    c.node("found?", "decision", "решение найдено?", x, 1170, 390, 145)
    c.node("show", "io", "вывод решения и метрик", 330, 1360, 430, 120)
    c.node("error", "io", "вывод сообщения о невозможности", 780, 1360, 430, 120)
    c.node("export?", "decision", "нужен экспорт CSV?", x, 1570, 390, 145)
    c.node("export", "io", "сохранение путей в CSV", 300, 1770, 430, 120)
    c.node("end", "terminal", "конец", x, 1950, 300, 92)
    c.arrow("start", "ui")
    c.arrow("ui", "task")
    c.arrow("task", "grid?")
    c.polyline([(c.nodes["grid?"].left, c.nodes["grid?"].y), (315, c.nodes["grid?"].y), (315, c.nodes["grid"].top)], "да")
    c.polyline([(c.nodes["grid?"].right, c.nodes["grid?"].y), (785, c.nodes["grid?"].y), (785, c.nodes["puzzle"].top)], "нет")
    c.polyline([(315, c.nodes["grid"].bottom), (315, 910), (550, 910), (550, c.nodes["metrics"].top)])
    c.polyline([(785, c.nodes["puzzle"].bottom), (785, 910), (550, 910), (550, c.nodes["metrics"].top)])
    c.arrow("metrics", "found?")
    c.polyline([(c.nodes["found?"].left, c.nodes["found?"].y), (330, c.nodes["found?"].y), (330, c.nodes["show"].top)], "да")
    c.polyline([(c.nodes["found?"].right, c.nodes["found?"].y), (780, c.nodes["found?"].y), (780, c.nodes["error"].top)], "нет")
    c.polyline([(330, c.nodes["show"].bottom), (330, 1480), (550, 1480), (550, c.nodes["export?"].top)])
    c.polyline([(780, c.nodes["error"].bottom), (780, 1480), (550, 1480), (550, c.nodes["export?"].top)])
    c.polyline([(c.nodes["export?"].left, c.nodes["export?"].y), (300, c.nodes["export?"].y), (300, c.nodes["export"].top)], "да")
    c.polyline([(300, c.nodes["export"].bottom), (300, 1870), (550, 1870), (550, c.nodes["end"].top)])
    c.polyline([(550, c.nodes["export?"].bottom), (550, c.nodes["end"].top)], "нет")
    c.save("01_main_program.png")


def draw_grid_dfs() -> None:
    c = Flowchart(1750)
    x = 550
    c.node("start", "subprocess", "запуск DFS", x, 80, 430, 115)
    c.node("graph", "process", "построить граф свободных клеток", x, 230)
    c.node("init", "process", "добавить старт в путь", x, 380)
    c.node("full?", "decision", "путь полный?", x, 560, 390, 140)
    c.node("finish?", "decision", "текущая клетка = финиш?", 300, 760, 390, 145)
    c.node("save", "process", "сохранить путь", 300, 960, 370, 105)
    c.node("neighbors", "process", "получить непосещенных соседей", 780, 760, 430, 115)
    c.node("has?", "decision", "есть сосед?", 780, 950, 350, 130)
    c.node("add", "process", "добавить соседа и продолжить DFS", 780, 1140, 440, 120)
    c.node("pop", "process", "удалить клетку из пути", 780, 1320, 390, 105)
    c.node("end?", "decision", "все ветви проверены?", x, 1490, 400, 135)
    c.node("out", "io", "вывести пути и метрики", x, 1650, 420, 105)
    c.arrow("start", "graph")
    c.arrow("graph", "init")
    c.arrow("init", "full?")
    c.polyline([(c.nodes["full?"].left, c.nodes["full?"].y), (300, c.nodes["full?"].y), (300, c.nodes["finish?"].top)], "да")
    c.polyline([(c.nodes["full?"].right, c.nodes["full?"].y), (780, c.nodes["full?"].y), (780, c.nodes["neighbors"].top)], "нет")
    c.arrow("finish?", "save", "да")
    c.polyline([(c.nodes["finish?"].right, c.nodes["finish?"].y), (510, c.nodes["finish?"].y), (510, 1410), (550, 1410), (550, c.nodes["end?"].top)], "нет")
    c.arrow("neighbors", "has?")
    c.arrow("has?", "add", "да")
    c.arrow("add", "pop")
    c.polyline([(780, c.nodes["pop"].bottom), (1000, c.nodes["pop"].bottom), (1000, 560), (c.nodes["full?"].right, 560)])
    c.polyline([(c.nodes["has?"].left, c.nodes["has?"].y), (550, c.nodes["has?"].y), (550, c.nodes["end?"].top)], "нет")
    c.polyline([(300, c.nodes["save"].bottom), (300, 1410), (550, 1410), (550, c.nodes["end?"].top)])
    c.arrow("end?", "out", "да")
    c.save("02_grid_dfs_backtracking.png")


def draw_grid_warnsdorff() -> None:
    c = Flowchart(1540)
    x = 550
    c.node("start", "subprocess", "запуск эвристики Варнсдорфа", x, 80, 500, 120)
    c.node("graph", "process", "построить граф и добавить старт", x, 240)
    c.node("cands", "process", "получить непосещенных соседей", x, 400)
    c.node("degree", "process", "вычислить число дальнейших ходов", x, 560)
    c.node("sort", "process", "отсортировать кандидатов", x, 720)
    c.node("has?", "decision", "есть кандидат?", x, 900, 360, 130)
    c.node("step", "process", "добавить лучшего кандидата", 320, 1100, 420, 110)
    c.node("full?", "decision", "путь полный и финиш?", 320, 1290, 390, 140)
    c.node("back", "process", "возврат и следующий кандидат", 780, 1100, 420, 110)
    c.node("out", "io", "вывести результат и метрики", x, 1450, 420, 110)
    for a, b in [("start", "graph"), ("graph", "cands"), ("cands", "degree"), ("degree", "sort"), ("sort", "has?")]:
        c.arrow(a, b)
    c.polyline([(c.nodes["has?"].left, c.nodes["has?"].y), (320, c.nodes["has?"].y), (320, c.nodes["step"].top)], "да")
    c.arrow("step", "full?")
    c.polyline([(c.nodes["full?"].right, c.nodes["full?"].y), (780, c.nodes["full?"].y), (780, c.nodes["back"].bottom)], "нет")
    c.polyline([(780, c.nodes["back"].top), (980, c.nodes["back"].top), (980, 900), (c.nodes["has?"].right, 900)])
    c.polyline([(320, c.nodes["full?"].bottom), (320, 1390), (550, 1390), (550, c.nodes["out"].top)], "да")
    c.polyline([(c.nodes["has?"].bottom[0] if False else 550, c.nodes["has?"].bottom), (550, c.nodes["out"].top)], "нет")
    c.save("03_grid_warnsdorff.png")


def draw_grid_connectivity() -> None:
    c = Flowchart(1660)
    x = 550
    c.node("start", "subprocess", "запуск отсечения по связности", x, 80, 500, 120)
    c.node("check", "process", "проверить связность, паритет, старт и финиш", x, 250, 540, 120)
    c.node("ok?", "decision", "условия выполнены?", x, 440, 390, 135)
    c.node("fail", "io", "вывести невозможность решения", 830, 620, 420, 115)
    c.node("choose", "process", "выбрать непосещенного соседа", 315, 620, 430, 115)
    c.node("residual", "process", "построить остаточный граф", 315, 800, 420, 110)
    c.node("connected?", "decision", "остаток связен и без конфликтов?", 315, 1000, 430, 155)
    c.node("prune", "process", "отсечь ветвь и вернуться", 770, 1000, 420, 110)
    c.node("dfs", "process", "продолжить рекурсивный поиск", 315, 1210, 430, 110)
    c.node("full?", "decision", "полный путь в финише?", 315, 1400, 390, 140)
    c.node("out", "io", "вывести пути и метрики", x, 1570, 420, 105)
    c.arrow("start", "check")
    c.arrow("check", "ok?")
    c.polyline([(c.nodes["ok?"].right, c.nodes["ok?"].y), (830, c.nodes["ok?"].y), (830, c.nodes["fail"].top)], "нет")
    c.polyline([(c.nodes["ok?"].left, c.nodes["ok?"].y), (315, c.nodes["ok?"].y), (315, c.nodes["choose"].top)], "да")
    c.arrow("choose", "residual")
    c.arrow("residual", "connected?")
    c.polyline([(c.nodes["connected?"].right, c.nodes["connected?"].y), (770, c.nodes["connected?"].y), (770, c.nodes["prune"].top)], "нет")
    c.polyline([(c.nodes["prune"].top + 0, c.nodes["prune"].bottom), (950, c.nodes["prune"].bottom), (950, 620), (c.nodes["choose"].right, 620)])
    c.arrow("connected?", "dfs", "да")
    c.arrow("dfs", "full?")
    c.polyline([(315, c.nodes["full?"].bottom), (315, 1510), (550, 1510), (550, c.nodes["out"].top)], "да")
    c.polyline([(c.nodes["full?"].right, c.nodes["full?"].y), (990, c.nodes["full?"].y), (990, 620), (c.nodes["choose"].right, 620)], "нет")
    c.polyline([(830, c.nodes["fail"].bottom), (830, 1510), (550, 1510), (550, c.nodes["out"].top)])
    c.save("04_grid_connectivity_pruning.png")


def draw_grid_backjumping() -> None:
    c = Flowchart(1580)
    x = 550
    c.node("start", "subprocess", "запуск Backjumping", x, 80, 430, 115)
    c.node("init", "process", "построить граф и добавить старт", x, 240)
    c.node("conflict?", "decision", "есть структурный конфликт?", x, 430, 430, 145)
    c.node("conflict", "process", "определить конфликтующие глубины", 270, 650, 430, 115)
    c.node("jump", "process", "прыжок к нужной глубине", 270, 840, 400, 110)
    c.node("cands", "process", "упорядочить кандидатов по Варнсдорфу", 800, 650, 440, 120)
    c.node("recurse", "process", "добавить клетку и продолжить поиск", 800, 840, 440, 120)
    c.node("success?", "decision", "путь полный и финиш?", x, 1080, 410, 145)
    c.node("save", "process", "сохранить решение", 300, 1300, 380, 105)
    c.node("next", "process", "следующий кандидат или прыжок назад", 780, 1300, 450, 110)
    c.node("out", "io", "вывести результат и метрики", x, 1490, 420, 105)
    c.arrow("start", "init")
    c.arrow("init", "conflict?")
    c.polyline([(c.nodes["conflict?"].left, c.nodes["conflict?"].y), (270, c.nodes["conflict?"].y), (270, c.nodes["conflict"].top)], "да")
    c.arrow("conflict", "jump")
    c.polyline([(270, c.nodes["jump"].bottom), (110, c.nodes["jump"].bottom), (110, 430), (c.nodes["conflict?"].left, 430)])
    c.polyline([(c.nodes["conflict?"].right, c.nodes["conflict?"].y), (800, c.nodes["conflict?"].y), (800, c.nodes["cands"].top)], "нет")
    c.arrow("cands", "recurse")
    c.polyline([(800, c.nodes["recurse"].bottom), (800, 980), (550, 980), (550, c.nodes["success?"].top)])
    c.polyline([(c.nodes["success?"].left, c.nodes["success?"].y), (300, c.nodes["success?"].y), (300, c.nodes["save"].top)], "да")
    c.polyline([(c.nodes["success?"].right, c.nodes["success?"].y), (780, c.nodes["success?"].y), (780, c.nodes["next"].top)], "нет")
    c.polyline([(780, c.nodes["next"].bottom), (980, c.nodes["next"].bottom), (980, 650), (c.nodes["cands"].right, 650)])
    c.polyline([(300, c.nodes["save"].bottom), (300, 1410), (550, 1410), (550, c.nodes["out"].top)])
    c.polyline([(780, c.nodes["next"].bottom), (780, 1410), (550, 1410), (550, c.nodes["out"].top)])
    c.save("05_grid_backjumping.png")


def draw_bfs() -> None:
    c = Flowchart(1550)
    x = 550
    c.node("start", "subprocess", "запуск BFS", x, 80, 390, 110)
    c.node("solvable", "decision", "конфигурация решаема?", x, 250, 390, 135)
    c.node("queue", "process", "поместить старт в очередь", 300, 460, 410, 105)
    c.node("empty?", "decision", "очередь пуста?", 300, 650, 350, 130)
    c.node("pop", "process", "извлечь первое состояние", 300, 830, 410, 105)
    c.node("goal?", "decision", "это цель?", 300, 1010, 320, 125)
    c.node("gen", "process", "сгенерировать допустимые ходы", 760, 830, 420, 115)
    c.node("visited?", "decision", "состояние посещено?", 760, 1030, 380, 135)
    c.node("push", "process", "сохранить родителя и добавить в очередь", 760, 1230, 470, 120)
    c.node("restore", "io", "восстановить путь и вывести решение", 300, 1230, 440, 120)
    c.node("fail", "io", "вывести отсутствие решения", 760, 460, 420, 105)
    c.node("end", "terminal", "конец", x, 1450, 300, 90)
    c.arrow("start", "solvable")
    c.polyline([(c.nodes["solvable"].left, c.nodes["solvable"].y), (300, c.nodes["solvable"].y), (300, c.nodes["queue"].top)], "да")
    c.polyline([(c.nodes["solvable"].right, c.nodes["solvable"].y), (760, c.nodes["solvable"].y), (760, c.nodes["fail"].top)], "нет")
    c.arrow("queue", "empty?")
    c.arrow("empty?", "pop", "нет")
    c.arrow("pop", "goal?")
    c.polyline([(c.nodes["goal?"].left, c.nodes["goal?"].y), (300, c.nodes["restore"].top)], "да")
    c.polyline([(c.nodes["goal?"].right, c.nodes["goal?"].y), (760, c.nodes["goal?"].y), (760, c.nodes["gen"].bottom)], "нет")
    c.arrow("gen", "visited?")
    c.arrow("visited?", "push", "нет")
    c.polyline([(760, c.nodes["push"].bottom), (960, c.nodes["push"].bottom), (960, 650), (c.nodes["empty?"].right, 650)])
    c.polyline([(c.nodes["visited?"].left, c.nodes["visited?"].y), (550, c.nodes["visited?"].y), (550, 650), (c.nodes["empty?"].right, 650)], "да")
    c.polyline([(c.nodes["empty?"].right, c.nodes["empty?"].y), (960, c.nodes["empty?"].y), (960, c.nodes["fail"].bottom)], "да")
    c.polyline([(300, c.nodes["restore"].bottom), (300, 1380), (550, 1380), (550, c.nodes["end"].top)])
    c.polyline([(760, c.nodes["fail"].bottom), (760, 1380), (550, 1380), (550, c.nodes["end"].top)])
    c.save("06_puzzle_bfs.png")


def draw_astar() -> None:
    c = Flowchart(1540)
    x = 550
    c.node("start", "subprocess", "запуск A*", x, 80, 360, 105)
    c.node("check", "decision", "конфигурация решаема?", x, 250, 390, 135)
    c.node("init", "process", "вычислить h и добавить в очередь", 300, 470, 440, 115)
    c.node("empty?", "decision", "очередь пуста?", 300, 660, 350, 130)
    c.node("pop", "process", "извлечь состояние с min f", 300, 840, 420, 105)
    c.node("goal?", "decision", "это цель?", 300, 1020, 320, 125)
    c.node("gen", "process", "сгенерировать соседей", 760, 840, 390, 105)
    c.node("better?", "decision", "путь до соседа лучше?", 760, 1030, 390, 135)
    c.node("update", "process", "обновить g, f и родителя", 760, 1230, 430, 115)
    c.node("out", "io", "восстановить оптимальный путь", 300, 1230, 430, 115)
    c.node("fail", "io", "вывести невозможность", 760, 470, 380, 105)
    c.node("end", "terminal", "конец", x, 1450, 300, 90)
    c.arrow("start", "check")
    c.polyline([(c.nodes["check"].left, c.nodes["check"].y), (300, c.nodes["check"].y), (300, c.nodes["init"].top)], "да")
    c.polyline([(c.nodes["check"].right, c.nodes["check"].y), (760, c.nodes["check"].y), (760, c.nodes["fail"].top)], "нет")
    c.arrow("init", "empty?")
    c.arrow("empty?", "pop", "нет")
    c.arrow("pop", "goal?")
    c.polyline([(c.nodes["goal?"].left, c.nodes["goal?"].y), (300, c.nodes["out"].top)], "да")
    c.polyline([(c.nodes["goal?"].right, c.nodes["goal?"].y), (760, c.nodes["goal?"].y), (760, c.nodes["gen"].bottom)], "нет")
    c.arrow("gen", "better?")
    c.arrow("better?", "update", "да")
    c.polyline([(760, c.nodes["update"].bottom), (960, c.nodes["update"].bottom), (960, 660), (c.nodes["empty?"].right, 660)])
    c.polyline([(c.nodes["better?"].left, c.nodes["better?"].y), (550, c.nodes["better?"].y), (550, 660), (c.nodes["empty?"].right, 660)], "нет")
    c.polyline([(c.nodes["empty?"].right, c.nodes["empty?"].y), (960, c.nodes["empty?"].y), (960, c.nodes["fail"].bottom)], "да")
    c.polyline([(300, c.nodes["out"].bottom), (300, 1380), (550, 1380), (550, c.nodes["end"].top)])
    c.polyline([(760, c.nodes["fail"].bottom), (760, 1380), (550, 1380), (550, c.nodes["end"].top)])
    c.save("07_puzzle_astar.png")


def draw_ida() -> None:
    c = Flowchart(1600)
    x = 550
    c.node("start", "subprocess", "запуск IDA*", x, 80, 380, 105)
    c.node("check", "decision", "конфигурация решаема?", x, 250, 390, 135)
    c.node("bound", "process", "установить bound = h(start)", 300, 470, 430, 110)
    c.node("dfs", "process", "поиск в глубину с g + h <= bound", 300, 650, 470, 120)
    c.node("goal?", "decision", "цель найдена?", 300, 850, 340, 130)
    c.node("cut?", "decision", "g + h > bound?", 760, 650, 360, 130)
    c.node("remember", "process", "запомнить минимальное превышение", 760, 850, 450, 115)
    c.node("children", "process", "проверить допустимых соседей", 300, 1060, 430, 110)
    c.node("more?", "decision", "решение найдено?", 300, 1240, 350, 130)
    c.node("newbound", "process", "увеличить bound и повторить", 760, 1240, 420, 110)
    c.node("out", "io", "вывести оптимальное решение", x, 1450, 430, 110)
    c.arrow("start", "check")
    c.polyline([(c.nodes["check"].left, c.nodes["check"].y), (300, c.nodes["check"].y), (300, c.nodes["bound"].top)], "да")
    c.polyline([(c.nodes["check"].right, c.nodes["check"].y), (900, c.nodes["check"].y), (900, c.nodes["out"].top)], "нет")
    c.arrow("bound", "dfs")
    c.arrow("dfs", "goal?")
    c.polyline([(c.nodes["goal?"].right, c.nodes["goal?"].y), (760, c.nodes["goal?"].y), (760, c.nodes["cut?"].top)], "нет")
    c.arrow("cut?", "remember", "да")
    c.polyline([(c.nodes["cut?"].left, c.nodes["cut?"].y), (300, c.nodes["cut?"].y), (300, c.nodes["children"].top)], "нет")
    c.arrow("children", "more?")
    c.polyline([(c.nodes["more?"].right, c.nodes["more?"].y), (760, c.nodes["more?"].y), (760, c.nodes["newbound"].top)], "нет")
    c.polyline([(760, c.nodes["newbound"].bottom), (950, c.nodes["newbound"].bottom), (950, 650), (c.nodes["dfs"].right, 650)])
    c.polyline([(c.nodes["goal?"].left, c.nodes["goal?"].y), (120, c.nodes["goal?"].y), (120, 1380), (550, 1380), (550, c.nodes["out"].top)], "да")
    c.polyline([(c.nodes["more?"].left, c.nodes["more?"].y), (120, c.nodes["more?"].y), (120, 1380), (550, 1380), (550, c.nodes["out"].top)], "да")
    c.save("08_puzzle_ida_star.png")


def draw_puzzle_backjumping() -> None:
    c = Flowchart(1580)
    x = 550
    c.node("start", "subprocess", "запуск Backjumping", x, 80, 420, 105)
    c.node("check", "decision", "конфигурация решаема?", x, 250, 390, 135)
    c.node("bound", "process", "установить границу по Manhattan", 300, 470, 450, 110)
    c.node("dfs", "process", "поиск в глубину с оценкой g + h", 300, 650, 450, 120)
    c.node("over?", "decision", "оценка выше границы?", 760, 650, 390, 135)
    c.node("jump", "process", "отсечь ветвь и вернуть новую границу", 760, 850, 470, 120)
    c.node("goal?", "decision", "состояние целевое?", 300, 850, 360, 130)
    c.node("sort", "process", "упорядочить ходы по Manhattan", 300, 1050, 440, 115)
    c.node("next", "process", "проверить кандидатов рекурсивно", 300, 1240, 440, 115)
    c.node("repeat", "process", "перейти к следующей границе", 760, 1240, 420, 110)
    c.node("out", "io", "вывести оптимальное решение", x, 1450, 430, 110)
    c.arrow("start", "check")
    c.polyline([(c.nodes["check"].left, c.nodes["check"].y), (300, c.nodes["check"].y), (300, c.nodes["bound"].top)], "да")
    c.polyline([(c.nodes["check"].right, c.nodes["check"].y), (900, c.nodes["check"].y), (900, c.nodes["out"].top)], "нет")
    c.arrow("bound", "dfs")
    c.polyline([(c.nodes["dfs"].right, c.nodes["dfs"].y), (760, c.nodes["dfs"].y), (760, c.nodes["over?"].top)])
    c.arrow("over?", "jump", "да")
    c.polyline([(760, c.nodes["jump"].bottom), (960, c.nodes["jump"].bottom), (960, c.nodes["repeat"].top)])
    c.polyline([(c.nodes["over?"].left, c.nodes["over?"].y), (300, c.nodes["over?"].y), (300, c.nodes["goal?"].top)], "нет")
    c.arrow("goal?", "sort", "нет")
    c.arrow("sort", "next")
    c.polyline([(c.nodes["next"].right, c.nodes["next"].y), (760, c.nodes["next"].y), (760, c.nodes["repeat"].top)], "нет")
    c.polyline([(760, c.nodes["repeat"].bottom), (950, c.nodes["repeat"].bottom), (950, 650), (c.nodes["dfs"].right, 650)])
    c.polyline([(c.nodes["goal?"].left, c.nodes["goal?"].y), (120, c.nodes["goal?"].y), (120, 1380), (550, 1380), (550, c.nodes["out"].top)], "да")
    c.save("09_puzzle_backjumping.png")


def clean_chain(
    name: str,
    nodes: list[tuple[str, str, str, int, int]],
    arrows: list[tuple[str, str, str]] | None = None,
    loops: list[tuple[str, str, str, str]] | None = None,
    branches: list[tuple[str, str, str, str]] | None = None,
    height: int = 1750,
) -> None:
    c = Flowchart(height)
    for node_name, kind, text, y, h in nodes:
        width = 520 if kind != "decision" else 470
        if kind == "terminal":
            width = 320
        c.node(node_name, kind, text, 550, y, width, h)
    if arrows:
        for start, end, label in arrows:
            c.arrow(start, end, label)
    if loops:
        for start, end, label, side in loops:
            a = c.nodes[start]
            b = c.nodes[end]
            x = 1000 if side == "right" else 100
            sx = a.right if side == "right" else a.left
            ex = b.right if side == "right" else b.left
            c.polyline([(sx, a.y), (x, a.y), (x, b.y), (ex, b.y)])
    if branches:
        for start, end, label, side in branches:
            a = c.nodes[start]
            b = c.nodes[end]
            x = 1000 if side == "right" else 100
            sx = a.right if side == "right" else a.left
            c.polyline([(sx, a.y), (x, a.y), (x, b.top), (b.x, b.top)])
    c.save(name)


def draw_grid_dfs_clean() -> None:
    nodes = [
        ("start", "subprocess", "запуск DFS", 70, 110),
        ("graph", "process", "построить граф свободных клеток", 220, 110),
        ("init", "process", "добавить старт в путь", 370, 110),
        ("full", "decision", "путь полный?", 540, 135),
        ("finish", "decision", "текущая клетка = финиш?", 720, 140),
        ("save", "process", "сохранить найденный путь", 900, 110),
        ("neighbors", "process", "получить непосещенных соседей", 1060, 110),
        ("has", "decision", "есть доступный сосед?", 1230, 135),
        ("step", "process", "добавить соседа и продолжить DFS", 1410, 115),
        ("pop", "process", "удалить клетку при возврате", 1570, 110),
        ("out", "io", "вывести пути и метрики", 1730, 105),
    ]
    arrows = [
        ("start", "graph", ""),
        ("graph", "init", ""),
        ("init", "full", ""),
        ("full", "finish", "да"),
        ("finish", "save", "да"),
        ("save", "neighbors", ""),
        ("neighbors", "has", ""),
        ("has", "step", "да"),
        ("step", "pop", ""),
        ("pop", "out", "все ветви"),
    ]
    loops = [("pop", "full", "возврат", "right")]
    branches = [
        ("full", "neighbors", "нет", "right"),
        ("finish", "out", "нет", "left"),
        ("has", "out", "нет", "left"),
    ]
    clean_chain("02_grid_dfs_backtracking.png", nodes, arrows, loops, branches, 1850)


def draw_grid_warnsdorff_clean() -> None:
    nodes = [
        ("start", "subprocess", "запуск эвристики Варнсдорфа", 70, 110),
        ("graph", "process", "построить граф и добавить старт", 220, 110),
        ("cands", "process", "получить непосещенных соседей", 370, 110),
        ("degree", "process", "вычислить число дальнейших ходов", 520, 110),
        ("sort", "process", "отсортировать кандидатов", 670, 110),
        ("has", "decision", "есть кандидат?", 840, 130),
        ("step", "process", "выбрать кандидата с минимальной степенью", 1020, 115),
        ("full", "decision", "путь полный и финиш?", 1200, 140),
        ("back", "process", "возврат и следующий кандидат", 1380, 110),
        ("out", "io", "вывести результат и метрики", 1540, 105),
    ]
    arrows = [
        ("start", "graph", ""),
        ("graph", "cands", ""),
        ("cands", "degree", ""),
        ("degree", "sort", ""),
        ("sort", "has", ""),
        ("has", "step", "да"),
        ("step", "full", ""),
        ("full", "out", "да"),
        ("back", "out", "все кандидаты"),
    ]
    loops = [("back", "has", "следующий", "right")]
    branches = [("has", "out", "нет", "left"), ("full", "back", "нет", "right")]
    clean_chain("03_grid_warnsdorff.png", nodes, arrows, loops, branches, 1660)


def draw_grid_connectivity_clean() -> None:
    nodes = [
        ("start", "subprocess", "запуск отсечения по связности", 70, 110),
        ("check", "process", "проверить связность, паритет, старт и финиш", 230, 115),
        ("ok", "decision", "начальные условия выполнены?", 410, 145),
        ("choose", "process", "выбрать непосещенного соседа", 600, 110),
        ("residual", "process", "построить остаточный граф", 760, 110),
        ("good", "decision", "остаток связен и без конфликтов?", 950, 150),
        ("dfs", "process", "продолжить рекурсивный поиск", 1150, 110),
        ("full", "decision", "полный путь в финише?", 1330, 140),
        ("prune", "process", "отсечь ветвь и вернуться", 1510, 110),
        ("out", "io", "вывести пути и метрики", 1680, 105),
    ]
    arrows = [
        ("start", "check", ""),
        ("check", "ok", ""),
        ("ok", "choose", "да"),
        ("choose", "residual", ""),
        ("residual", "good", ""),
        ("good", "dfs", "да"),
        ("dfs", "full", ""),
        ("full", "out", "да"),
        ("prune", "out", "все ветви"),
    ]
    loops = [("prune", "choose", "следующая ветвь", "right")]
    branches = [
        ("ok", "out", "нет", "left"),
        ("good", "prune", "нет", "right"),
        ("full", "choose", "нет", "left"),
    ]
    clean_chain("04_grid_connectivity_pruning.png", nodes, arrows, loops, branches, 1800)


def draw_grid_backjumping_clean() -> None:
    nodes = [
        ("start", "subprocess", "запуск Backjumping", 70, 110),
        ("init", "process", "построить граф и добавить старт", 220, 110),
        ("conflict", "decision", "есть структурный конфликт?", 400, 145),
        ("set", "process", "определить конфликтующие глубины", 590, 115),
        ("jump", "process", "выполнить прыжок назад", 750, 110),
        ("cands", "process", "упорядочить кандидатов по Варнсдорфу", 920, 120),
        ("step", "process", "добавить клетку и продолжить поиск", 1100, 115),
        ("success", "decision", "путь полный и финиш?", 1280, 140),
        ("next", "process", "следующий кандидат или прыжок назад", 1460, 115),
        ("out", "io", "вывести результат и метрики", 1630, 105),
    ]
    arrows = [
        ("start", "init", ""),
        ("init", "conflict", ""),
        ("conflict", "set", "да"),
        ("set", "jump", ""),
        ("cands", "step", ""),
        ("step", "success", ""),
        ("success", "out", "да"),
        ("next", "out", "все ветви"),
    ]
    loops = [("jump", "conflict", "к глубине", "left"), ("next", "cands", "продолжить", "right")]
    branches = [("conflict", "cands", "нет", "right"), ("success", "next", "нет", "right")]
    clean_chain("05_grid_backjumping.png", nodes, arrows, loops, branches, 1750)


def draw_bfs_clean() -> None:
    nodes = [
        ("start", "subprocess", "запуск BFS", 70, 110),
        ("check", "decision", "конфигурация решаема?", 240, 140),
        ("queue", "process", "поместить старт в очередь", 430, 110),
        ("empty", "decision", "очередь пуста?", 600, 130),
        ("pop", "process", "извлечь первое состояние", 780, 110),
        ("goal", "decision", "это цель?", 950, 125),
        ("gen", "process", "сгенерировать допустимые ходы", 1120, 110),
        ("visited", "decision", "состояние посещено?", 1290, 135),
        ("push", "process", "сохранить родителя и добавить в очередь", 1470, 120),
        ("out", "io", "восстановить путь или вывести отказ", 1660, 115),
    ]
    arrows = [
        ("start", "check", ""),
        ("check", "queue", "да"),
        ("queue", "empty", ""),
        ("empty", "pop", "нет"),
        ("pop", "goal", ""),
        ("goal", "gen", "нет"),
        ("gen", "visited", ""),
        ("visited", "push", "нет"),
        ("push", "out", "если цель"),
    ]
    loops = [("push", "empty", "следующее состояние", "right")]
    branches = [
        ("check", "out", "нет", "left"),
        ("empty", "out", "да", "left"),
        ("goal", "out", "да", "left"),
        ("visited", "empty", "да", "right"),
    ]
    clean_chain("06_puzzle_bfs.png", nodes, arrows, loops, branches, 1780)


def draw_astar_clean() -> None:
    nodes = [
        ("start", "subprocess", "запуск A*", 70, 110),
        ("check", "decision", "конфигурация решаема?", 240, 140),
        ("init", "process", "вычислить h и добавить старт в очередь", 430, 120),
        ("empty", "decision", "очередь пуста?", 610, 130),
        ("pop", "process", "извлечь состояние с минимальным f", 790, 110),
        ("goal", "decision", "это цель?", 960, 125),
        ("gen", "process", "сгенерировать соседние состояния", 1130, 110),
        ("better", "decision", "найден путь лучше?", 1300, 135),
        ("update", "process", "обновить g, f и родителя", 1480, 110),
        ("out", "io", "восстановить оптимальный путь", 1660, 110),
    ]
    arrows = [
        ("start", "check", ""),
        ("check", "init", "да"),
        ("init", "empty", ""),
        ("empty", "pop", "нет"),
        ("pop", "goal", ""),
        ("goal", "gen", "нет"),
        ("gen", "better", ""),
        ("better", "update", "да"),
        ("update", "out", "если цель"),
    ]
    loops = [("update", "empty", "следующее min f", "right")]
    branches = [
        ("check", "out", "нет", "left"),
        ("empty", "out", "да", "left"),
        ("goal", "out", "да", "left"),
        ("better", "empty", "нет", "right"),
    ]
    clean_chain("07_puzzle_astar.png", nodes, arrows, loops, branches, 1780)


def draw_ida_clean() -> None:
    nodes = [
        ("start", "subprocess", "запуск IDA*", 70, 110),
        ("check", "decision", "конфигурация решаема?", 240, 140),
        ("bound", "process", "установить bound = h(start)", 430, 110),
        ("dfs", "process", "поиск в глубину с g + h <= bound", 600, 120),
        ("goal", "decision", "цель найдена?", 780, 130),
        ("over", "decision", "g + h > bound?", 960, 130),
        ("remember", "process", "запомнить минимальное превышение", 1140, 115),
        ("children", "process", "проверить допустимых соседей", 1310, 110),
        ("newbound", "process", "увеличить bound и повторить", 1480, 110),
        ("out", "io", "вывести оптимальное решение", 1660, 110),
    ]
    arrows = [
        ("start", "check", ""),
        ("check", "bound", "да"),
        ("bound", "dfs", ""),
        ("dfs", "goal", ""),
        ("goal", "over", "нет"),
        ("over", "remember", "да"),
        ("remember", "children", ""),
        ("children", "newbound", "если нет решения"),
        ("newbound", "out", "если цель"),
    ]
    loops = [("newbound", "dfs", "новая граница", "right")]
    branches = [
        ("check", "out", "нет", "left"),
        ("goal", "out", "да", "left"),
        ("over", "children", "нет", "right"),
    ]
    clean_chain("08_puzzle_ida_star.png", nodes, arrows, loops, branches, 1780)


def draw_puzzle_backjumping_clean() -> None:
    nodes = [
        ("start", "subprocess", "запуск Backjumping", 70, 110),
        ("check", "decision", "конфигурация решаема?", 240, 140),
        ("bound", "process", "установить границу по Manhattan", 430, 110),
        ("dfs", "process", "поиск в глубину с оценкой g + h", 600, 120),
        ("over", "decision", "оценка выше границы?", 780, 135),
        ("jump", "process", "отсечь ветвь и вернуть новую границу", 970, 120),
        ("goal", "decision", "состояние целевое?", 1160, 135),
        ("sort", "process", "упорядочить ходы по Manhattan", 1350, 110),
        ("next", "process", "проверить кандидатов рекурсивно", 1520, 110),
        ("out", "io", "вывести оптимальное решение", 1690, 110),
    ]
    arrows = [
        ("start", "check", ""),
        ("check", "bound", "да"),
        ("bound", "dfs", ""),
        ("dfs", "over", ""),
        ("over", "jump", "да"),
        ("jump", "goal", "новая граница"),
        ("goal", "sort", "нет"),
        ("sort", "next", ""),
        ("next", "out", "если цель"),
    ]
    loops = [("next", "dfs", "следующая граница", "right")]
    branches = [
        ("check", "out", "нет", "left"),
        ("over", "goal", "нет", "right"),
        ("goal", "out", "да", "left"),
    ]
    clean_chain("09_puzzle_backjumping.png", nodes, arrows, loops, branches, 1810)


def main() -> None:
    draw_main()
    draw_grid_dfs_clean()
    draw_grid_warnsdorff_clean()
    draw_grid_connectivity_clean()
    draw_grid_backjumping_clean()
    draw_bfs_clean()
    draw_astar_clean()
    draw_ida_clean()
    draw_puzzle_backjumping_clean()


if __name__ == "__main__":
    main()
