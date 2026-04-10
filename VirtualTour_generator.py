# virtual_tour_generator.py

import turtle
import textwrap
import subprocess
from pathlib import Path

import re, json, os
from datetime import datetime
from typing import Any, Dict, Dict, List, Tuple, Union
from typing import List
import numpy as np
from sklearn.metrics import cluster
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import colorsys
from turtle import Screen, Turtle, done, title

# ==================================================
#        قسم الجولات الافتراضية اللولبية
# ==================================================
class VirtualTour_Pipeline:
    """Pipeline لتحليل النصوص بتقنية Helix + Virtual Tours"""

    # Class variables
    width: int = 1400
    height: int = 1000

    def __init__(self, sim_threshold: float = 0.9, max_depth: int = 15) -> None:
        """إعداد الجولات الافتراضية اللولبية"""
        print("🚀 تهيئة VirtualTour Pipeline...")

        # 1. إعدادات
        self.sim_threshold: float = sim_threshold
        self.max_depth: int = max_depth

        # 2. أدوات وبيانات
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(max_features=100)
        self.insights: Dict[str, Union[List[str], int, List[int]]] = defaultdict(list)

        # 3. حالة الجولات
        self.spiral_analysis: List[str] = []
        self.tour_code: str = ""

        # 4. تقارير ونتائج
        self.verified_reports: List[str] = []
        self.test_metrics: List[Dict[str, Any]] = []
        self.reports: List[str] = []

        # 5. حالة النظام
        self.is_clean: bool = True

        print(f"✅ جاهز | threshold={sim_threshold}, max_depth={max_depth}")

    def helix_puzzle_analyzer(self, puzzle_text: str, max_lines: int = 20) -> str:
        """
        تقنية Helix: تجزئة الألغاز/المقالات لخطوط ملونة DNA-style

        Args:
            puzzle_text: النص المعقد (لغز/مقالة/بحث)
            max_lines: أقصى عدد خطوط (افتراضي 20)

        Returns:
            كود turtle جاهز للتنفيذ
        """
        print(f"🧬 تحليل Helix: '{puzzle_text[:60]}...'")

        # 1. تجزئة ذكية للجمل المنطقية
        sentences = re.split(r'[.،؟!؛\n]+', puzzle_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        sentences = sentences[:max_lines]

        print(f"   → {len(sentences)} خطوط مكتشفة")

        # 2. ألوان Helix (DNA double helix)
        colors = ['red', 'blue', 'green', 'orange', 'purple',
                'yellow', 'pink', 'brown', 'cyan', 'magenta']

        # 3. بناء كود Turtle
        helix_code = textwrap.dedent(f"""\
        import turtle
        import math

        screen = turtle.Screen()
        screen.setup(1600, 900)
        screen.bgcolor('black')
        screen.title('🧬 Helix Text Analyzer - VirtualTour Pipeline')
        screen.tracer(0, 0)  # سرعة عالية

        t = turtle.Turtle()
        t.speed(0)
        t.hideturtle()
        t.pensize(4)

        # Helix Functions
        def draw_helix_line(color_idx, y_pos, text, line_num):
            t.penup()
            t.goto(-650, y_pos)
            t.pendown()
            t.pencolor(colors[color_idx])
            t.write(f"خط {{line_num}}: {{text[:55]}}...",
                    font=("Arial", 12, "bold"), align="left")

            # Helix Spiral للخط
            t.pencolor('white')
            t.penup(); t.goto(-550, y_pos)
            t.pendown()
            t.pensize(2)
            for i in range(45):
                angle = i * 8
                radius = 15 + math.sin(i * 0.3) * 5
                t.setheading(angle)
                t.forward(radius)

        # الخطوط الملونة
        colors = {colors}
        y_offset = 120
        for i, sentence in enumerate({sentences}):
            draw_helix_line(i % len(colors), y_offset - i * 45, sentence, i+1)

        # DNA Helix Connector (ربط الخطوط)
        t.pensize(3)
        t.pencolor('yellow')
        t.penup(); t.goto(-300, 80); t.pendown()
        for i in range(360 * 2):
            x = 300 + 200 * math.cos(i * math.pi / 180)
            y = 80 + 30 * math.sin(i * math.pi / 60)
            t.goto(x, y)

        # Answer Indicator (رمز النتيجة)
        t.penup(); t.goto(400, -100); t.pendown()
        t.fillcolor('orange'); t.pencolor('red')
        t.begin_fill()
        for _ in range(5):  # نجمة خماسية
            t.forward(50); t.right(144)
        t.end_fill()
        t.color('black')
        t.write("النتيجة", font=("Arial", 18, "bold"))

        # إحصائيات
        t.penup(); t.goto(-650, -250)
        t.pendown(); t.pencolor('lightgray')
        t.write(f"إجمالي الخطوط: {{len({sentences})}} | Helix Depth: {{len({sentences})*3}}",
                font=("Arial", 14, "normal"))

        screen.update()
        screen.exitonclick()
        """)

        # إحصائيات التحليل
        print(f"   → Helix Depth: {len(sentences)*3}")
        print(f"   → كود جاهز: {len(helix_code)} حرف")

        return helix_code

    def line_manager(self, colored_lines: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        إدارة ذكية للخطوط الملونة

        Args:
            colored_lines: [{'text': '...', 'color': 'red', 'id': 1}, ...]

        Returns:
            {'clusters': [...], 'connections': [...], 'summary': '...'}
        """

        # 1. تجميع الخطوط المتشابهة (Clustering)
        clusters = self._cluster_similar_lines(colored_lines)

        # 2. ربط الخطوط المنطقية (Connections)
        connections = self._connect_logical_flow(clusters)

        # 3. تلخيص الهيكل
        summary = self._generate_structure_summary(connections)

        return {
            'clusters': clusters,
            'connections': connections,
            'line_count': len(colored_lines),
            'cluster_count': len(clusters),
            'summary': summary
        }

    def _cluster_similar_lines(self, lines: List[Dict]) -> List[List[Dict]]:
        """تجميع خطوط متشابهة"""
        # TF-IDF + KMeans للتجميع
        texts = [line['text'] for line in lines]
        vectors = self.vectorizer.fit_transform(texts).todense()

        # KMeans clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(5, len(lines)//3))
        clusters = kmeans.fit_predict(vectors)

        # تجميع حسب المجموعات
        result = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            result[cluster_id].append(lines[i])

        return list(result.values())

    def _connect_logical_flow(self, clusters: List) -> List[Dict]:
        """ربط المجموعات منطقيًا"""
        connections = []
        for i, cluster in enumerate(clusters):
            if i < len(clusters) - 1:
                connections.append({
                    'from': i,
                    'to': i+1,
                    'weight': self._calculate_connection_strength(cluster, clusters[i+1])
                })
        return connections

    def virtual_tour_generator(self, helix_lines: List[str], width: int = 1400, height: int = 1000, depth: int = 1) -> str:
        """
        جولات افتراضية لولبية: ربط خطوط Helix في نموذج بصري

        Args:
            helix_lines: خطوط ملونة من helix_text_analyzer
            width/height: أبعاد الشاشة
            depth: عمق الجولة الحالية

        Returns:
            كود turtle جاهز للـ visualization
        """
        print(f"🔄 جولة {depth}: ربط {len(helix_lines)} خطوط Helix")

        # -------- إعادة تعيين الحالة لجولة جديدة --------
        self.spiral_analysis: List[str] = helix_lines.copy()
        self.insights.clear()
        self.width = width
        self.height = height

        # -------- تحليل لولبي لخطوط Helix --------
        depth = min(depth, self.max_depth)
        print(f"\n--- جولة {depth}/{self.max_depth} ---")

        # 1. استخراج Insights من Helix lines
        all_text = " ".join(helix_lines).lower()
        shapes: List[str] = re.findall(r'(دائرة|مربع|مثلث|خط|دائرة)', all_text)
        colors: List[str] = re.findall(r'(أزرق|أحمر|أخضر|أصفر|برتقالي|بنفسجي)', all_text)
        nums: List[str] = re.findall(r'\d+', all_text)
        complexity: int = len(shapes) + len(colors) + len(nums) + depth

        # 2. بناء spiral_analysis
        new_insight = f"عمق{depth}: {len(helix_lines)} خطوط، شكل{shapes or 'غير محدد'}، تعقيد{complexity}"
        self.spiral_analysis.append(new_insight)

        # 3. حفظ Insights
        self.insights.update({
            'shapes': shapes,
            'colors': colors,
            'helix_lines': helix_lines,
            'line_count': [str(len(helix_lines))],
            'complexity': [str(complexity)],
            'depth': [str(depth)]
        })

        # 4. التحقق من التشابه (اختياري)
        if len(self.spiral_analysis) > 1:
            vectors: np.ndarray = np.asarray(
                self.vectorizer.fit_transform(self.spiral_analysis[-2:]).todense()
            )
            similarity: float = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            print(f"   تشابه Helix: {similarity:.3f}")

            if similarity > self.sim_threshold:
                print(f"❌ توقف: تشابه عالي ({similarity:.3f} > {self.sim_threshold})")

        # 4,5.استخراج القوائم مرة واحدة + تحويل آمن لـ int
        shapes = self.insights.get('shapes', [])
        colors = self.insights.get('colors', [])
        complexity_list = self.insights.get('complexity', ['10'])
        depth_list = self.insights.get('depth', ['1'])
        complexity = int(complexity_list[0]) if complexity_list else 10
        depth_val = int(depth_list[0]) if depth_list else 1

        # 5. تحديد المعاملات النهائية
        final_shape: str = shapes[-1] if shapes else 'دائرة'
        final_steps: int = min(150, 50 + depth_val * 8)
        complexity_val: int = complexity

        # 5. تحديد المعاملات النهائية
        final_shape: str = shapes[-1] if shapes else 'دائرة'
        final_steps: int = min(150, 50 + int(depth[0]) * 8) if isinstance(depth, list) else min(150, 50 + depth * 8)
        complexity_val: int = int(complexity[0]) if isinstance(complexity, list) else int(complexity)

        print(f"شكل: {final_shape} | خطوات: {final_steps} | تعقيد: {complexity_val}")

        # 6. Visual (متوافق مع Dict[str, List[str]])
        visual_insights = {
            'shapes': shapes,
            'colors': colors,
            'helix_lines': self.insights.get('helix_lines', []),
            'complexity': [str(complexity_val)],     # ✅ List[str]
            'steps': [str(final_steps)]              # ✅ List[str]
        }
        self.tour_code = self.generate_turtle_visual(visual_insights, final_steps)

        # إدارة الخطوط الملونة
        colored_lines = [{'text': line, 'color': colors[i % len(colors)] if colors else 'red'}
        for i, line in enumerate(helix_lines)]
        line_analysis = self.line_manager(colored_lines)
        self.insights['line_manager'] = line_analysis

        # Visual مع Clusters
        self.tour_code = self.generate_cluster_visual(line_analysis)

        # 7. التقرير النهائي
        print(f"✅ اكتملت الجولة {depth}")
        print(f"   Insights: {dict(self.insights)}")
        print(f"   جولات إجمالية: {len(self.spiral_analysis)}")
        print("✅ كود جاهز | exec(self.tour_code)")

        return self.tour_code

    def get_insights(self):
        """استرجاع نتائج الجولات"""
        return dict(self.insights)

    def spiral_study(self, idea: str) -> Tuple[int, Dict[str, List[str]]]:
        """الجولات اللولبية"""
        self.spiral_analysis = [idea]
        depth: int = 0

        while depth < self.max_depth:
            depth += 1
            prev = self.spiral_analysis[-1].lower()
            shapes = re.findall(r'(دائرة|مربع|مثلث)', prev)
            colors = re.findall(r'(أزرق|أحمر|أخضر|أصفر)', prev)
            complexity = len(shapes) + len(colors) + depth

            new_insight = f"عمق{depth}: شكل{shapes or '?'}، تعقيد{complexity}"
            self.spiral_analysis.append(new_insight)
            self.insights['shapes'].extend(shapes)
            self.insights['colors'].extend(colors)

            if len(self.spiral_analysis) > 1:
                vectors = np.asarray(self.vectorizer.fit_transform(self.spiral_analysis[-2:]).todense())
                sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                if sim > self.sim_threshold:
                    break  # خارج الحلقة

        return depth, dict(self.insights)  # ✅ دايمًا يرجع!

    def generate_turtle_visual(self, insights: Dict[str, List[str]], depth: int) -> str:
        """
        دالة مخصصة لتوليد كود Turtle بصري واضح
        """
        final_shape = insights['shapes'][-1] if insights['shapes'] else 'دائرة'
        final_steps = min(150, 50 + depth * 8)
        return self._build_news_detector_visual(final_shape, final_steps, depth)

    def generate_cluster_visual(self, line_analysis: Dict[str, Any]) -> str:
        """توليد كود بصري لعرض المجموعات والاتصالات"""
        clusters = line_analysis.get('clusters', [])
        connections = line_analysis.get('connections', [])

        # -------- تحويل آمن لـ f-string وإحصائيات ---------------------
        cluster_sizes = [len(c) for c in clusters]
        safe_connections = [{'from': c['from'], 'to': c['to'], 'weight': getattr(c, 'weight', 0.5)}
                            for c in connections[:5]]

        screen = turtle.Screen()
        screen.setup(1400, 1000)
        screen.bgcolor('black')
        screen.title('📊 Cluster Visualization - Virtual Tour Helix')
        screen.tracer(0)

        t = turtle.Turtle()
        t.speed(0)
        t.hideturtle()
        t.pensize(4)

        # ألوان Helix
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

        # -------- رسم المجموعات (مربعات ملونة) -----------------------
        num_clusters = min(5, len({cluster_sizes}))
        for i in range(num_clusters):
            x = -600 + i * 280
            t.penup()
            t.goto(x, 250)
            t.pendown()
            t.fillcolor(colors[i])
            t.pencolor('white')
            t.pensize(3)
            t.begin_fill()
            for _ in range(4):
                t.forward(220)
                t.right(90)
            t.end_fill()
            # نص الـ cluster
            t.color('black')
            t.goto(x + 110, 230)
            t.write(f'Cluster {i+1}\\n{len({clusters}[i])} خطوط',
                    font=("Arial", 14, "bold"), align="center")

        # -------- رسم الاتصالات (أقواس منحنية) -----------------------
        t.pensize(4)
        for conn in {safe_connections}:
            from_idx = conn['from']
            to_idx = conn['to']
            weight = conn['weight']
            t.pencolor('yellow' if weight > 0.7 else 'orange')
            t.pensize(2 + weight * 4)

            # نقطة البداية
            start_x = -600 + from_idx * 280 + 110
            end_x = -600 + to_idx * 280 + 110

            t.penup()
            t.goto(start_x, 250)
            t.pendown()
            # قوس منحني بدل خط مستقيم
            dist = end_x - start_x
            t.setheading(0)
            t.circle(60 * abs(dist)/200, 180)  # قوس جميل
            t.goto(end_x, 250)

            # وزن الاتصال
            mid_x = (start_x + end_x) / 2
            t.penup()
            t.goto(mid_x, 280)
            t.pendown()
            t.color('white')
            t.write(f'w={weight:.2f}', font=("Arial", 12, "bold"), align="center")

        # -------- إحصائيات Helix والمجموعات -----------------------
        t.penup()
        t.goto(0, -200)
        t.pensize(2)
        t.pencolor('cyan')
        t.write(f'Clusters: {num_clusters} | Connections: {len({safe_connections})} | Helix Lines: {sum({cluster_sizes})}',
                font=("Arial", 16, "bold"))

        screen.update()
        screen.exitonclick()

        return textwrap.dedent(f'''\
        # كود Cluster Visualization جاهز
        # Clusters: {num_clusters} | Connections: {len({safe_connections})} | Helix Lines: {sum({cluster_sizes})}
        ''')

    def _build_news_detector_visual(self, shape: str, steps: int, depth: int) -> str:
        """بناء النموذج البصري لـ News Detector"""
        return textwrap.dedent(f"""\
        import turtle

        screen = turtle.Screen()
        screen.setup(1400, 1000)
        screen.bgcolor('black')
        screen.title('📰 News Detector - Virtual Tour')

        t = turtle.Turtle()
        t.speed(0)
        t.hideturtle()
        t.pensize(4)

        # 1️⃣ NLP Analysis Pipeline (مستطيل أحمر)
        t.penup(); t.goto(-300, 100); t.pendown()
        t.fillcolor('red'); t.pencolor('darkred')
        t.begin_fill()
        for _ in range(4): t.forward(120); t.right(90)
        t.end_fill()
        t.write("NLP", font=("Arial", 16, "bold"))

        # 2️⃣ Fact-Check DB (مثلث أزرق)
        t.penup(); t.goto(0, 100); t.pendown()
        t.fillcolor('blue'); t.pencolor('darkblue')
        t.begin_fill()
        for _ in range(3): t.forward(100); t.right(120)
        t.end_fill()
        t.write("Fact DB", font=("Arial", 16, "bold"))

        # 3️⃣ Real-time Verify (دائرة خضراء)
        t.penup(); t.goto(300, 100); t.pendown()
        t.fillcolor('green'); t.pencolor('darkgreen')
        t.begin_fill(); t.circle(60); t.end_fill()
        t.write("Verify", font=("Arial", 16, "bold"))

        # 4️⃣ Accuracy Flow (أسهم صفراء)
        t.pencolor('yellow'); t.pensize(3)
        t.penup(); t.goto(-180, 50); t.pendown()
        t.setheading(-30); t.forward(80)
        t.penup(); t.goto(-80, 50); t.pendown()
        t.setheading(30); t.forward(80)
        t.penup(); t.goto(80, 50); t.pendown()
        t.setheading(150); t.forward(80)

        # 5️⃣ Accuracy Badge
        t.penup(); t.goto(0, -150); t.pendown()
        t.fillcolor('gold'); t.pencolor('orange')
        t.begin_fill(); t.circle(40); t.end_fill()
        t.color('black'); t.write("95%", font=("Arial", 20, "bold"))

        screen.exitonclick()
        """)

    def type_txt_reporter(self, arg):
        """تقرير TXT"""
        os.makedirs('output/reports', exist_ok=True)
        report_path = f"output/reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"تقرير - {arg}\n")
            f.write("=" * 40 + "\n")
            f.write("تفاصيل...\n")

        if not hasattr(self, 'reports'):
            self.reports = []
        self.reports.append(report_path)
        return report_path

    def verification_tour(self, report_path: str) -> Tuple[str, List[str]]:
        """
        جولة تدقيق واحدة: قراءة TXT، تحليل، تصحيح.
        """
        print("\n🔍 جولة التدقيق الواحدة...")

        content = ""
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            return "", ["❌ خطأ: الملف غير موجود"]  # str فارغ بدل None

        except Exception as e:
            return "", [f"❌ خطأ: {str(e)}"]

        # -------- تحليل آمن (كودك كما هو) --------
        lines = content.split('\n')
        depth_match = next((re.search(r'عدد الجولات:\s*(\d+)', line) for line in lines if 'جولات' in line), None)
        depth = int(depth_match.group(1)) if depth_match else 0

        sim_match = re.search(r'التشابه النهائي:\s*([\d.]+)', content)
        similarity = float(sim_match.group(1)) if sim_match else 0

        # -------- insights بسيط وآمن --------
        insights = {'shapes': ['دائرة']}

        issues = []
        if similarity < 0.5:
            issues.append("تحذير: تغييرات كبيرة")

        verified_depth = min(depth, 12)
        verified_report = f"{report_path}_verified.txt"

        with open(verified_report, 'w', encoding='utf-8') as f:
            f.write(f"تقرير تدقيق - العمق: {verified_depth}\n")
            f.write(f"التشابه: {similarity}\n")
            f.write("التصحيحات:\n" + "\n".join(f"- {issue}" for issue in issues) + "\n")
            f.write(json.dumps(insights, ensure_ascii=False, indent=2) + "\n✅ تم التحقق!\n")

        print(f"تم التدقيق: {verified_report}")
        return verified_report, issues  # ✅ يطابق الـ type hint

    def test_tour(self, verified_report: str, temp_code_file: str = 'temp_test.py') -> Tuple[Dict[str, Any], Union[str, None]]:
        """
        جولة اختبار: تنفيذ، قياس، تقرير.
        """
        print("\n🚀 جولة الاختبار...")

        try:
            with open(verified_report, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            return {'error': 'report not found'}, None  # ✅ يشتغل الآن!

        # -------- استخراج عمق --------
        depth_match = re.search(r'عمق:\s*(\d+)', content)
        depth = int(depth_match.group(1)) if depth_match else 10

        # -------- كود اختبار مُصحح --------
        test_code = f"""from turtle import *
    import time
    start = time.time()
    screen = Screen()
    screen.setup(800, 600)
    t = Turtle()
    t.speed(0)
    for i in range({depth}):
        t.circle(50 + i*2)
        t.right(5)
    end = time.time()
    print(f"TIME={{end-start:.2f}} DEPTH={{depth}}")
    screen.exitonclick()
    """

        Path(temp_code_file).write_text(test_code)

        # -------- تنفيذ --------
        try:
            result = subprocess.run(['python', temp_code_file],
                                capture_output=True, text=True, timeout=10)
            test_passed = result.returncode == 0

            time_match = re.search(r'TIME=([\d.]+)', result.stdout)
            exec_time = float(time_match.group(1)) if time_match else 0

            metrics = {'exec_time': exec_time, 'depth': depth, 'passed': test_passed}

        except subprocess.TimeoutExpired:
            metrics = {'error': 'timeout', 'passed': False}

        # -------- تقرير --------
        test_report = verified_report.replace('.txt', '_test.txt')
        with open(test_report, 'w', encoding='utf-8') as f:
            f.write(f"حالة: {'✅ نجح' if metrics['passed'] else '❌ فشل'}\n")
            f.write(f"Metrics: {{metrics}}\n")

        print(f"نتيجة: {{metrics['passed']}} | {test_report}")
        return metrics, test_report

    def _save_report(self, name: str):
        """حفظ تقرير"""
        os.makedirs('output/reports', exist_ok=True)
        report_path = f"output/reports/{name}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"تقرير {name}\n")
            f.write(f"الجولات: {self.spiral_analysis}\n")
        self.reports.append(report_path)
        return report_path

    def cleanup(self) -> None:
        """تنظيف شامل للحالة والملفات"""
        print("🧹 تنظيف Pipeline...")

        # تنظيف الحالة
        self.spiral_analysis.clear()
        self.insights.clear()
        self.verified_reports.clear()
        self.test_metrics.clear()
        self.tour_code = ""
        self.is_clean = True

        # حذف الملفات المؤقتة
        for file_pattern in ['temp_test.py', '*_test.txt', '*_verified.txt']:
            for file_path in Path('.').glob(file_pattern):
                try:
                    file_path.unlink()
                    print(f"   🗑️  {file_path.name}")
                except:
                    pass

        # إغلاق turtle
        try:
            turtle.bye()
        except:
            pass

        print("✅ تنظيف مكتمل")

    def full_analysis_pipeline(self, complex_idea: str) -> str:
        """Pipeline كامل: Helix → Virtual Tour → Report → Test"""

        # 1️⃣ دراسة الفكر (فصل خطوط)
        helix_lines = self.helix_text_analyzer(complex_idea)

        # 2️⃣ تشكيل الفكرة (ربط + loop)
        tour_code = ""
        depth = 0
        while depth < self.max_depth:
            tour_code = self.virtual_tour_generator(helix_lines)
            depth, insights = self.spiral_study(helix_lines)  # التحقق

            if depth >= self.max_depth:
                break

        # 3️⃣ تقرير
        report_path = self.type_txt_reporter(tour_code, insights)

        # 4️⃣ اختبار نهائي
        metrics, test_report = self.test_tour(report_path)

        return f"✅ تحليل مكتمل | عمق: {depth} | دقة: {metrics}"

if __name__ == "__main__":
    """مثال: كشف أخبار حقيقية vs مزيفة"""
    pipeline = VirtualTour_Pipeline()

    # طلب المستخدم المعقد
    news_detector_idea = """AI يكشف الأخبار الحقيقية من المزيفة:
تحليل نصوص، sentiment analysis،
fact-checking database، NLP models،
real-time verification، accuracy 95%،
معالجة 1000 خبر/ثانية، معقد جدًا"""

    # ✅ خطوة 1: استخراج Helix lines أولاً
    helix_code = pipeline.helix_puzzle_analyzer(news_detector_idea, max_lines=20)
    print("🧬 Helix code جاهز!")

    # ✅ خطوة 2: استخراج الخطوط الفعلية (split النص)
    sentences = re.split(r'[.،؟!؛\n]+', news_detector_idea)
    helix_lines = [s.strip() for s in sentences if len(s.strip()) > 5][:20]
    print(f"📝 {len(helix_lines)} خطوط Helix مستخرجة")

    # ✅ خطوة 3: الآن virtual tour
    code = pipeline.virtual_tour_generator(helix_lines, width=1400, height=1000)

    print("\n" + "="*80)
    print("📰 كود AI كشف الأخبار مولّد!")
    print(f"Insights: {pipeline.get_insights()}")
    print(f"جولات: {len(pipeline.spiral_analysis)}")

    exec(code)  # تشغيل الجولة المرئية

    pipeline.cleanup()
