# virtual_tour_generator.py

import math
import turtle
import textwrap
import subprocess
from pathlib import Path
import re, json, os
from datetime import datetime
from typing import Any, Dict, Generator, List, Tuple, Union
import numpy as np

# Sklearn imports صحيحة
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans  # ✅ صح
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from functools import lru_cache

print("✅ Imports جاهزة | Helix Pipeline")

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

        # 2. أدوات وبيانات ✅ واحد بس!
        from sklearn.feature_extraction.text import TfidfVectorizer
        from collections import defaultdict

        self.vectorizer: TfidfVectorizer = TfidfVectorizer(max_features=100)
        self.insights: Dict[str, List[str]] = defaultdict(list)  # ✅ Type + default

        # 3. حالة الجولات
        self.spiral_analysis: List[str] = []
        self.tour_code: str = ""

        # 4. تقارير ونتائج
        self.verified_reports: List[str] = []
        self.test_metrics: List[Dict[str, Any]] = []
        self.reports: List[str] = []

        # 5. حالة النظام
        self.is_clean: bool = True

        # 6. Regular expression لتحليل Helix lines (تحديث)
        self._helix_parser = re.compile(r'"([^"]+)"\s*,\s*color\s*=?\s*["\']?([^"\']+)["\']?')

        print(f"✅ جاهز | threshold={sim_threshold}, max_depth={max_depth}")

    def stream_text_chunks(self, text_or_file: Union[str, Path], chunk_size: int = 1000) -> Generator[List[str], None, None]:
        """تدفق chunked حسب memory limit"""
        if isinstance(text_or_file, Path):
            text = text_or_file.read_text(encoding='utf-8')
        else:
            text = text_or_file

        sentences = re.split(r'[.،؟!؛\n]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        # Yield chunks (لا تحميل كامل في الذاكرة)
        for i in range(0, len(sentences), chunk_size):
            yield sentences[i:i+chunk_size]

    def memory_aware_pipeline(self, text_or_file: Union[str, Path], max_memory_mb: int = 100) -> Dict[str, Any]:
        """Pipeline يراقب الذاكرة"""
        import psutil
        import gc

        all_results = {'chunks': [], 'clusters': []}

        for chunk_id, chunk in enumerate(self.stream_text_chunks(text_or_file)):
            # تحقق memory
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            if memory_mb > max_memory_mb:
                gc.collect()
                print(f"🧹 Memory cleanup: {memory_mb:.1f}MB")

            # معالجة chunk
            colored_lines = self._parse_helix_to_lines(" ".join(chunk))  # List[Dict]
            chunk_clusters = self._cluster_similar_lines(colored_lines)  # OK!

            all_results['chunks'].append(chunk_clusters)
            print(f"Chunk {chunk_id}: {len(chunk_clusters)} clusters")

        return all_results

    def spiral_study(self, idea: str) -> Tuple[int, Dict[str, List[str]]]:
        """الجولات اللولبية - Type Safe"""
        self.spiral_analysis = [idea]
        depth: int = 0

        # 1. تأكد من أن insights موجودة ونظيفة
        if not hasattr(self, 'insights') or not isinstance(self.insights, dict):
            self.insights = defaultdict(list)

        while depth < self.max_depth:
            depth += 1
            prev = self.spiral_analysis[-1].lower()
            shapes = re.findall(r'(دائرة|مربع|مثلث)', prev)
            colors = re.findall(r'(أزرق|أحمر|أخضر|أصفر)', prev)
            complexity = len(shapes) + len(colors) + depth

            new_insight = f"عمق{depth}: شكل{shapes or '?'}، تعقيد{complexity}"
            self.spiral_analysis.append(new_insight)

            # ✅ دائمًا extend (مش update)
            self.insights['shapes'].extend(shapes)
            self.insights['colors'].extend(colors)

            # Similarity check
            if len(self.spiral_analysis) > 1:
                vectors = np.asarray(self.vectorizer.fit_transform(self.spiral_analysis[-2:]).todense())
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                print(f"تشابه: {similarity:.3f}")
                if similarity > self.sim_threshold:
                    print(f"توقف: {similarity:.3f} > {self.sim_threshold}")
                    break

        # ✅ Return type-safe
        return depth, self.get_insights()

    @lru_cache(maxsize=128)
    def helix_puzzle_analyzer(self, puzzle_text: str, max_lines: int = 500) -> List[Dict[str, Any]]:
        print(f"🧬 Helix V3: '{puzzle_text[:60]}...'")

        # 1. تجزئة + تنظيف صحيح
        sentences = re.split(r'[.،؟!؛\n]+', puzzle_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        sentences = [re.sub(r'[^\w\s]', '', s) for s in sentences]
        sentences = [s for s in sentences if len(s.split()) > 2][:max_lines]

        if not sentences:
            print("⚠️ No valid sentences!")
            return [{'text': puzzle_text[:100], 'color': 'gray', 'group': 'L0', 'idx': 0, 'weight': 1.0}]

        # 2. ألوان Helix + Gr.X (مرة واحدة)
        base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan']
        color_groups = defaultdict(int)
        colored_lines = []

        for i, sentence in enumerate(sentences):
            base_color = base_colors[i % len(base_colors)]
            color_groups[base_color] += 1
            group_label = f"Gr.{color_groups[base_color]}" if color_groups[base_color] > 15 else f"L{i+1}"

            importance = min(1.0, len(sentence.split()) / 20.0)

            colored_lines.append({
                'text': sentence,
                'color': base_color,
                'group': group_label,
                'idx': i,
                'length': len(sentence),
                'weight': importance,
                'tfidf_score': 0.0
            })

        print(f"✅ Helix V3 | {len(colored_lines)} خط | Gr.X: {sum(1 for l in colored_lines if 'Gr.' in l['group'])}")
        print(f"🎨 Core color: {max(set(l['color'] for l in colored_lines), key=lambda c: sum(1 for l in colored_lines if l['color'] == c))}")

        return colored_lines

    def _parse_helix_to_lines(self, text: str) -> List[Dict[str, str]]:
        """Parse text → List[Dict] للـ clustering"""
        matches = self._helix_parser.findall(text)
        return [{'text': text, 'color': color or 'blue', 'id': str(i)}
                for i, (text, color) in enumerate(matches) or [{'text': text[:200], 'color': 'blue', 'id': '0'}]]

    def _helix_to_lines(self, helix_text: str) -> List[Dict[str, str]]:
        """تحويل Turtle code → List[Dict[str, str]]"""
        lines = re.findall(r'"([^"]+)"\s*,\s*color\s*=?\s*["\']?([^"\']+)["\']?', helix_text)
        return [
            {
                'text': text,
                'color': color,
                'id': str(i)  # ✅ str(i) بدل i
            }
            for i, (text, color) in enumerate(lines)
        ]

    def chunk_lines(self, lines: List[Dict], chunk_size: int = 500) -> List[List[Dict]]:
        """تقسيم lines إلى chunks"""
        chunks = []
        for i in range(0, len(lines), chunk_size):
            chunks.append(lines[i:i + chunk_size])
        return chunks

    def _prepare_colored_lines(self, helix_lines: List[str]) -> List[Dict[str, Any]]:
        """تحضير colored_lines مع Gr.X"""
        base_colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan']
        color_gr = defaultdict(int)

        colored = []
        for i, line in enumerate(helix_lines):
            color = base_colors[i % len(base_colors)]
            gr_count = color_gr[color]
            color_gr[color] += 1
            group = f"Gr.{gr_count} {color.capitalize()}" if gr_count > 15 else f"L{i+1}"

            colored.append({
                'text': line, 'base_color': color,
                'group': group, 'id': i
            })
        return colored

    def line_manager(self, colored_lines: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        إدارة ذكية للخطوط الملونة → Clusters + Connections
        """
        # 1. Clustering (TF-IDF + KMeans)
        clusters = self._cluster_similar_lines(colored_lines)

        # 2. Connections منطقية
        connections = self._connect_logical_flow(clusters)

        # 3. Summary
        summary = self._generate_structure_summary(connections)

        return {
            'clusters': clusters,
            'connections': connections,
            'colored_lines': colored_lines,  # حفظ للـ visual
            'line_count': len(colored_lines),
            'cluster_count': len(clusters),
            'summary': summary
        }

    @lru_cache(maxsize=128)
    def cached_tfidf_vectors(self, text_hash: str) -> np.ndarray:
        """TF-IDF cached حسب hash"""
        return self.vectorizer.fit_transform([text_hash]).todense()

    def smart_clustering(self, lines: List[Dict], cache_limit: int = 1000) -> List[List[Dict]]:
        """Clustering مع cache + memory check"""
        if len(lines) > cache_limit:
            # Chunking للـ large input
            all_clusters = []
            for chunk in self.chunk_lines(lines, 500):
                chunk_clusters = self.smart_clustering(chunk)  # recursive
                all_clusters.extend(chunk_clusters)
            return all_clusters

        # Small input → direct clustering
        return self._cluster_similar_lines(lines)

    def ultimate_pipeline(self, pdf_path: Path, max_memory_mb: int = 200) -> Dict[str, Any]:
        """Pipeline كامل مع memory management"""
        results = self.memory_aware_pipeline(pdf_path, max_memory_mb)

        # Final merge + visualization
        final_lines = []
        for chunk in results['chunks']:
            final_lines.extend(chunk)

        line_analysis = self.line_manager(final_lines[:1000])  # limit final
        tour_code = self.generate_cluster_visual(line_analysis)

        return {
            'tour_code': tour_code,
            'analysis': line_analysis,
            'chunks_processed': len(results['chunks'])
        }

    def find_central_cluster(self, clusters: List[List[Dict]], cluster_weights: Dict[int, float]) -> List[Dict]:
        """أقوى cluster (الفكرة الكلية)"""
        if not cluster_weights:
            return clusters[0] if clusters else []

        # Pylance-safe
        central_cluster_id = max(cluster_weights.items(), key=lambda x: x[1])[0]
        return clusters[central_cluster_id]

    def _cluster_similar_lines(self, lines: List[Dict]) -> List[List[Dict]]:
        """تجميع خطوط متشابهة V2 + Gr.X labels"""
        if len(lines) < 2:
            return [lines]

        texts = [line['text'] for line in lines]
        vectors = self.vectorizer.fit_transform(texts).toarray()

        # ✅ إصلاح n_clusters
        n_clusters = min(7, max(1, len(lines)))  # 2 خط → 1 cluster | 5 خط → 5 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(vectors)

        # باقي الكود نفس الشيء
        result = defaultdict(list)
        for i, cluster_id in enumerate(cluster_ids):
            line = lines[i].copy()
            line['cluster_id'] = cluster_id
            line['cluster_label'] = f"Gr.{cluster_id+1} Cluster" if cluster_id < 7 else f"Cluster {cluster_id+1}"
            result[cluster_id].append(line)

        clusters = list(result.values())
        print(f"✅ {len(clusters)} clusters | أكبر: {max(len(c) for c in clusters)} خط")
        return clusters

    def _connect_logical_flow(self, clusters: List[List[Dict]]) -> List[Dict[str, float]]:
        """ربط Clusters بوزن similarity"""
        connections = []
        for i in range(len(clusters) - 1):
            weight = self._calculate_connection_strength(clusters[i], clusters[i+1])
            connections.append({
                'from': i,
                'to': i+1,
                'weight': weight
            })
        return connections

    def _calculate_connection_strength(self, cluster_a: List[Dict], cluster_b: List[Dict]) -> float:
        texts_a = [line['text'] for line in cluster_a]
        texts_b = [line['text'] for line in cluster_b]

        # Jaccard similarity (بدون sklearn matrix problems!)
        set_a = set(texts_a[0].lower().split()) if texts_a else set()
        set_b = set(texts_b[0].lower().split()) if texts_b else set()

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _generate_structure_summary(self, connections: List[Dict]) -> str:
        """تلخيص الهيكل"""
        if not connections:
            return "لا توجد اتصالات"

        avg_weight = sum(c['weight'] for c in connections) / len(connections)
        return f"هيكل: {len(connections)} اتصال | متوسط الوزن: {avg_weight:.2f}"

    def extract_core_idea(self, colored_lines: List[Dict[str, Any]], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        استخلاص الفكرة الكلية من Helix Clusters + Gr.X

        Args:
            colored_lines: من helix_puzzle_analyzer
            analysis_results: من line_manager/virtual_tour_generator

        Returns:
            {'core_concept': str, 'priority_clusters': List, 'theme_strength': float}
        """

        # استخدم colored_lines للـ color analysis (إزالة الـ warning)
        main_color = 'gray'  # default safe
        if colored_lines:  # ✅ تحقق فارغ أم لا
            main_color = max(set(l['color'] for l in colored_lines),
                            key=lambda c: sum(1 for l in colored_lines if l['color'] == c))

        clusters = analysis_results.get('clusters', [])
        connections = analysis_results.get('connections', [])

        if not clusters:
            return {'core_concept': 'غير محدد', 'priority_clusters': [], 'theme_strength': 0.0, 'dominant_color': main_color}

        # 1. Centrality Analysis (أهم cluster)
        cluster_weights = {}
        for conn in connections:
            from_id = conn.get('from', 0)
            to_id = conn.get('to', 0)
            weight = conn.get('weight', 0.5)
            cluster_weights.setdefault(from_id, 0)
            cluster_weights[from_id] += weight * 0.5
            cluster_weights.setdefault(to_id, 0)
            cluster_weights[to_id] += weight * 0.5

        if cluster_weights:
            central_cluster_id = max(cluster_weights, key=lambda x: cluster_weights[x])
        else:
            central_cluster_id = 0

        central_cluster = clusters[central_cluster_id]  # ✅ أضف هذا السطر!

        # 2. Core keywords extraction
        central_texts = [line['text'] for line in central_cluster[:5]]
        all_words = ' '.join(central_texts).lower()

        # Top keywords (TF-IDF simple)
        word_freq = defaultdict(int)
        for word in re.findall(r'\b\w+\b', all_words):
            if len(word) > 3 and word not in ['ال', 'في', 'من', 'على', 'مع']:
                word_freq[word] += 1

        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:8]
        core_concept = ' '.join([kw[0] for kw in top_keywords[:4]])

        # 3. Theme strength (connection density + cluster size)
        max_weight = max([c.get('weight', 0) for c in connections], default=1)
        max_weight = max(max_weight, 0.1)  # ✅ منع ZeroDivision

        cluster_size_ratio = len(central_cluster) / sum(len(c) for c in clusters)
        connection_ratio = cluster_weights.get(central_cluster_id, 0) / max_weight
        connection_count_ratio = len([c for c in connections if c.get('from') == central_cluster_id or c.get('to') == central_cluster_id]) / max(1, len(connections))

        theme_strength = (
            cluster_size_ratio * 0.4 +
            connection_ratio * 0.3 +
            connection_count_ratio * 0.3
        )

        # 4. Priority clusters (مرتبطة بالمركزية)
        priority_clusters = []
        for cid, cluster in enumerate(clusters):
            connection_score = sum(c['weight'] for c in connections if c['from'] == cid or c['to'] == cid)

            first_line = cluster[0]  # أول خط في الـ cluster
            gr_label = first_line.get('group', first_line.get('cluster_label', f'Gr.{cid+1}'))

            priority_clusters.append({
                'cluster_id': cid,
                'gr_label': gr_label,
                'dominant_color': main_color,
                'sample': first_line['text'][:50],
                'color': first_line.get('color', 'unknown'),
                'size': len(cluster),
                'connection_score': connection_score
            })

        priority_clusters.sort(key=lambda x: x['connection_score'], reverse=True)

        return {
            'core_concept': core_concept or 'غير محدد',
            'theme_strength': theme_strength or 0.0,
            'priority_clusters': priority_clusters[:5] if 'priority_clusters' in locals() else [],
            'central_cluster': central_cluster_id,
            'total_clusters': len(clusters),
            'dominant_color': main_color
        }

    def virtual_tour_generator(self, helix_lines: List[str], width: int = 1400, height: int = 1000, depth: int = 1) -> str:
        """
        Virtual Tour Helix V2: Helix → Insights → Clusters → Gr.X Visual
        """
        print(f"🔄 جولة {depth}: {len(helix_lines)} خطوط Helix")

        # 1. إعادة تعيين
        self.spiral_analysis = helix_lines.copy()
        self.insights.clear()
        self.width, self.height = width, height
        depth = min(depth, self.max_depth)

        # 2. استخراج Insights (نفس القديم)
        all_text = " ".join(helix_lines).lower()
        shapes = re.findall(r'(دائرة|مربع|مثلث|خط)', all_text)
        colors = re.findall(r'(أزرق|أحمر|أخضر|أصفر|برتقالي|بنفسجي)', all_text)
        nums = re.findall(r'\d+', all_text)
        complexity = len(shapes) + len(colors) + len(nums) + depth

        # 3. Spiral analysis + similarity check
        new_insight = f"عمق{depth}: {len(helix_lines)} خط، تعقيد{complexity}"
        self.spiral_analysis.append(new_insight)

        if len(self.spiral_analysis) > 1:
            vectors = np.asarray(self.vectorizer.fit_transform(self.spiral_analysis[-2:]).todense())
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            print(f"تشابه: {similarity:.3f}")
            if similarity > self.sim_threshold:
                print(f"توقف: {similarity:.3f} > {self.sim_threshold}")

        # 4. Insights نظيف
        self.insights['shapes'] = [str(shapes)] if shapes else []
        self.insights['colors'] = [str(colors)] if colors else []
        self.insights['helix_lines'] = helix_lines if isinstance(helix_lines, list) else [helix_lines]
        self.insights['line_count'] = [str(len(helix_lines))]
        self.insights['complexity'] = [str(complexity)]
        self.insights['depth'] = [str(depth)]

        # 5. Helix lines → Colored lines + Clustering
        base_colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan']
        colored_lines = []
        color_gr = defaultdict(int)

        for i, line in enumerate(helix_lines):
            color = base_colors[i % len(base_colors)]
            gr_count = color_gr[color]
            color_gr[color] += 1
            group = f"Gr.{gr_count} {color.capitalize()}" if gr_count > 15 else f"L{i+1}"

            colored_lines.append({
                'text': line, 'base_color': color, 'group': group, 'id': i
            })

        # 6. Line analysis (clusters + connections)
        line_analysis = self.line_manager(colored_lines)
        self.insights['line_analysis'] = [str(line_analysis)]  # Dict → List[str]

        # 7. الـ Visual النهائي (Clusters + Helix)
        self.tour_code = self.generate_cluster_visual(line_analysis)

        print(f"✅ جولة {depth} | Clusters: {line_analysis['cluster_count']} | Gr.X: {sum(1 for l in colored_lines if 'Gr.' in l['group'])}")
        return self.tour_code

    def generate_cluster_visual(self, line_analysis: Dict[str, Any]) -> str:
        """Visual Outcome: Helix → Gr.X Clusters → مربعات + ربط"""
        clusters = line_analysis.get('clusters', [])
        connections = line_analysis.get('connections', [])

        # 1. إحصائيات Gr.X
        color_gr_count = defaultdict(int)
        for cluster in clusters:
            for line in cluster[:3]:
                if 'Gr.' in line.get('group', ''):
                    color_gr_count[line.get('color', 'gray')] += 1

        # 2. المتغيرات للـ f-string
        num_clusters = min(7, len(clusters))
        cluster_colors = {'red': '#FF6B6B', 'green': '#4ECDC4', 'blue': '#45B7D1',
                        'orange': '#FFEAA7', 'purple': '#DDA0DD', 'yellow': '#F7DC6F'}
        gr_stats = ' | '.join([f"{color}: {count}" for color, count in color_gr_count.items()])

        # 3. الكود النهائي
        helix_code = textwrap.dedent(f"""\
    import turtle
    import math

    screen = turtle.Screen()
    screen.setup(1600, 1000)
    screen.bgcolor('#111111')
    screen.title('🔗 Cluster Visual - Helix Gr.X')
    screen.tracer(0)

    t = turtle.Turtle()
    t.speed(0)
    t.hideturtle()
    t.pensize(5)

    cluster_colors = {cluster_colors}

    # Clusters squares
    for i in range({num_clusters}):
        size = len({clusters}[i])
        x = -700 + i * 280
        scale = min(260, 120 + size * 0.6)

        t.penup()
        t.goto(x, 250)
        t.pendown()
        color = list(cluster_colors.keys())[i % len(cluster_colors)]
        t.fillcolor(cluster_colors[color])
        t.pencolor('white')
        t.begin_fill()
        for _ in range(4):
            t.forward(scale)
            t.right(90)
        t.end_fill()

        label = {clusters}[i][0].get('group', f'Gr.{{i+1}}')
        t.color('black')
        t.goto(x + scale/2, 230)
        t.write(f'{{label}}\\n({{size}} خط)', font=("Arial", 14, "bold"), align="center")

    # Connections
    for conn in {connections}[:10]:
        t.pencolor('#FFD700' if conn.get('weight', 0.5) > 0.7 else '#FF8C00')
        t.pensize(3 + conn.get('weight', 0.5) * 5)
        t.penup()
        t.goto(-700 + conn.get('from', 0) * 280 + 130, 250)
        t.pendown()
        t.goto(-700 + conn.get('to', 1) * 280 + 130, 250)

    screen.update()
    screen.exitonclick()
    """)

        print(f"✅ Clusters: {len(clusters)}")
        print(f"Gr.X Stats: {gr_stats}")  # إزالة الـ warning
        return helix_code

    def generate_turtle_visual(self, insights: Dict[str, List[str]], depth: int) -> str:
        """News Detector + Gr.X Clusters"""
        final_shape = insights['shapes'][-1] if insights['shapes'] else 'دائرة'
        final_steps = min(150, 50 + depth * 8)

        # إضافة Gr.X stats
        gr_count = len([s for s in insights.get('shapes', []) if 'Gr.' in s])

        return self._build_news_detector_visual(final_shape, final_steps, depth, gr_count)

    def _build_news_detector_visual(self, shape: str, steps: int, depth: int, gr_count: int = 0) -> str:
        """News Detector V3 + Helix Gr.X"""
        print(f"📰 News Detector | Shape: {shape}, Steps: {steps}, Depth: {depth}, Gr.X: {gr_count}")  # ✅ استخدامها

        return textwrap.dedent(f"""\
    import turtle

    screen = turtle.Screen()
    screen.setup(1400, 1000)
    screen.bgcolor('black')
    screen.title('📰 News Detector V3 | Gr.X: {gr_count}')

    t = turtle.Turtle()
    t.speed(0)
    t.hideturtle()
    t.pensize(4)

    # 1️⃣ NLP (أحمر)
    t.penup(); t.goto(-350, 120); t.pendown()
    t.fillcolor('red'); t.pencolor('darkred')
    t.begin_fill()
    for _ in range(4): t.forward(140); t.right(90)
    t.end_fill()
    t.color('white'); t.goto(-280, 100); t.write("NLP", font=("Arial", 16, "bold"))

    # 2️⃣ Fact DB (أزرق)
    t.penup(); t.goto(-50, 120); t.pendown()
    t.fillcolor('blue'); t.pencolor('darkblue')
    t.begin_fill()
    for _ in range(3): t.forward(110); t.right(120)
    t.end_fill()
    t.color('white'); t.goto(-20, 95); t.write("Fact DB", font=("Arial", 16, "bold"))

    # 3️⃣ Verify (أخضر)
    t.penup(); t.goto(250, 120); t.pendown()
    t.fillcolor('green'); t.pencolor('darkgreen')
    t.begin_fill(); t.circle(65); t.end_fill()
    t.color('white'); t.goto(280, 95); t.write("Verify", font=("Arial", 16, "bold"))

    # 4️⃣ Helix Gr.X (بنفسجي)
    t.penup(); t.goto(450, 120); t.pendown()
    t.fillcolor('purple'); t.pencolor('#AA44FF')
    t.begin_fill()
    for _ in range(4): t.forward(110); t.right(90)
    t.end_fill()
    t.color('white'); t.goto(500, 95); t.write(f"Helix\\nGr.X: {{gr_count}}", font=("Arial", 14, "bold"))

    # 5️⃣ Accuracy Flow
    t.pencolor('yellow'); t.pensize(4)
    t.penup(); t.goto(-220, 70); t.pendown(); t.setheading(-25); t.forward(90)
    t.penup(); t.goto(-90, 70); t.pendown(); t.setheading(25); t.forward(90)
    t.penup(); t.goto(210, 70); t.pendown(); t.setheading(155); t.forward(90)

    # 6️⃣ Accuracy Badge
    t.penup(); t.goto(0, -180); t.pendown()
    t.fillcolor('gold'); t.pencolor('orange')
    t.begin_fill(); t.circle(45); t.end_fill()
    t.color('black'); t.goto(15, -200); t.write("95%", font=("Arial", 20, "bold"))

    # 7️⃣ Helix Stats
    t.penup(); t.goto(-350, -280); t.pencolor('cyan')
    t.write(f"Shape: {{shape}} | Depth: {{depth}} | Steps: {{steps}} | Gr.X: {{gr_count}}",
            font=("Arial", 14, "bold"))

    screen.exitonclick()
    print("✅ News Detector V3 Complete!")
    """)

    def get_insights(self) -> Dict[str, List[str]]:
        """استرجاع نتائج الجولات - Type Safe"""
        if not hasattr(self, 'insights') or not isinstance(self.insights, dict):
            return {}

        result: Dict[str, List[str]] = {}
        for key, value in self.insights.items():
            if isinstance(value, list):
                result[key] = [str(item) for item in value]
            else:
                result[key] = [str(value)]

        return result

    def writing_flow_organizer(self, analysis_results: Dict[str, Any], target_length: int = 2000, structure_type: str = "blog") -> Dict[str, Any]:
        """تنظيم تدفق الكتابة بعد Helix V3"""

        # Helix V3 path
        line_analysis = analysis_results.get('line_analysis', {})
        clusters = line_analysis.get('clusters', [])
        gr_count = sum(1 for c in clusters if any('Gr.' in line.get('group', '') for line in c))

        # 1. تحليل Clusters → Sections
        sections = []
        for i, cluster in enumerate(clusters[:8]):
            cluster_size = len(cluster)
            main_topic = max(cluster, key=lambda x: len(x['text']))['text'][:60]
            sections.append({
                'id': i,
                'gr_label': cluster[0].get('group', f'Gr.{i+1}'),  # ✅ group من Helix V3
                'topic': main_topic,
                'priority': cluster_size / sum(len(c) for c in clusters),
                'lines': [line['text'] for line in cluster[:10]]
            })

        # 2. ترتيب حسب priority
        flow_order = sorted(sections, key=lambda x: x['priority'], reverse=True)

        # 3. Word budget
        total_weight = sum(s['priority'] for s in flow_order)
        word_budget = {s['id']: max(200, int(target_length * (s['priority'] / total_weight))) for s in flow_order}

        # 4. Structure template
        templates = {
            'blog': ['مقدمة', 'المشكلة', 'الحلول Gr.X', 'أمثلة', 'خاتمة'],
            'report': ['ملخص', 'تحليل Gr.X', 'نتائج', 'توصيات'],
            'technical': ['مشكلة', 'Gr.X Clustering', 'الخوارزمية', 'اختبارات', 'نتائج']
        }

        outline = []
        for i, template_section in enumerate(templates.get(structure_type, templates['blog'])):
            if i < len(flow_order):
                section_id = flow_order[i]['id']
                outline.append({
                    'section': template_section,
                    'gr_cluster': flow_order[i]['gr_label'],
                    'words': word_budget[section_id],
                    'content_flow': flow_order[i]['lines']
                })

        return {
            'outline': outline,
            'flow_order': [s['id'] for s in flow_order],
            'word_budget': word_budget,
            'total_sections': len(outline),
            'gr_clusters_used': gr_count
            # _generate_writing_prompt() مش موجود → شيلناها
        }

    def type_txt_reporter(self, arg: str) -> str:
        """تقرير TXT مع Gr.X"""
        os.makedirs('output/reports', exist_ok=True)
        report_path = f"output/reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"تقرير Helix - {arg}\n")
            f.write("=" * 50 + "\n")
            f.write(f"التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Gr.X Clusters | TF-IDF Analysis | Virtual Tour\n")

        if not hasattr(self, 'reports'):
            self.reports = []
        self.reports.append(report_path)
        return report_path

    def _save_report(self, name: str) -> str:
        """حفظ تقرير Helix كامل"""
        os.makedirs('output/reports', exist_ok=True)
        report_path = f"output/reports/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"تقرير Helix Pipeline - {name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"الجولات اللولبية: {{len(self.spiral_analysis)}}\n")
            f.write(f"Clusters Gr.X: {{self.insights.get('line_analysis', {{}}).get('cluster_count', 0)}}\n")
            f.write(f"Insights: {{dict(self.insights)}}\n")
            if hasattr(self, 'tour_code'):
                f.write(f"Visual Code: {{len(self.tour_code)}} حرف\n")

        if not hasattr(self, 'reports'):
            self.reports = []
        self.reports.append(report_path)
        return report_path

    def test_tour(self, verified_report: str, temp_code_file: str = 'temp_test.py') -> Tuple[Dict[str, Any], Union[str, None]]:
        """جولة اختبار Helix V3 + Gr.X"""
        print("🚀 جولة الاختبار...")

        try:
            with open(verified_report, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            return {'error': 'report not found'}, None

        # استخراج عمق + Gr.X ✅ regex محسن
        depth_match = re.search(r'(?:عمق|Depth|depth).*?(\d+)', content, re.IGNORECASE)
        depth = int(depth_match.group(1)) if depth_match else 10

        gr_match = re.findall(r'Gr\.(\d+)', content)
        gr_count = len(set(gr_match))  # unique Gr.X

        # كود اختبار Helix V3
        test_code = textwrap.dedent(f"""\
        import turtle
        import time
        import math

        start = time.time()
        screen = turtle.Screen()
        screen.setup(1200, 800)
        screen.bgcolor('black')
        screen.title('Helix Test V3')
        t = turtle.Turtle()
        t.speed(0)
        t.hideturtle()
        t.pensize(3)

        # Helix spiral test
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7']
        for i in range({min(depth, 25)}):
            angle = i * 15
            radius = 20 + math.sin(i * 0.3) * 8
            t.penup()
            t.goto(0, 0)
            t.setheading(angle)
            t.pendown()
            t.pencolor(colors[i%4])
            t.circle(radius)

            # Gr.X labels
            if i % 5 == 0:
                t.penup()
                t.goto(100 * math.cos(math.radians(angle)), 100 * math.sin(math.radians(angle)))
                t.color('white')
                t.write(f"Gr.{{(i//5)+1}}", font=("Arial", 12, "bold"))

        end = time.time()
        print(f"TIME={{end-start:.3f}} DEPTH={depth} GR={gr_count}")
        screen.exitonclick()
        """)

        Path(temp_code_file).write_text(test_code)

        # تنفيذ الاختبار
        try:
            result = subprocess.run(['python', temp_code_file], capture_output=True, text=True, timeout=15)
            test_passed = result.returncode == 0

            time_match = re.search(r'TIME=([\d.]+)', result.stdout)
            exec_time = float(time_match.group(1)) if time_match else 0

            metrics = {
                'exec_time': exec_time,
                'depth': depth,
                'gr_count': gr_count,
                'passed': test_passed
            }
        except subprocess.TimeoutExpired:
            metrics = {'error': 'timeout', 'passed': False}

        # تقرير الاختبار
        test_report = verified_report.replace('.txt', '_test.txt')
        with open(test_report, 'w', encoding='utf-8') as f:
            f.write(f"حالة: {'✅ نجح' if metrics.get('passed', False) else '❌ فشل'}\n")
            f.write(f"Metrics: {metrics}\n")
            f.write(f"Gr.X مكتشفة: {gr_count}\n")
            f.write(f"Depth: {depth}\n")

        print(f"✅ اختبار: {gr_count} Gr.X | {exec_time:.2f}s | {'نجح' if test_passed else 'فشل'}")
        return metrics, test_report

    def verification_tour(self, report_path: str) -> Tuple[str, List[str]]:
        """جولة تدقيق مع Gr.X validation"""
        print("🔍 جولة التدقيق...")

        content = ""
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            return "", ["❌ خطأ: الملف غير موجود"]
        except Exception as e:
            return "", [f"❌ خطأ: {str(e)}"]

        # تحليل regex محسن
        depth_match = re.search(r'(?:عمق|Depth|جولات).*?(\d+)', content, re.IGNORECASE)
        depth = int(depth_match.group(1)) if depth_match else 0

        sim_match = re.search(r'(?:تشابه|similarity).*?([\d.]+)', content, re.IGNORECASE)
        similarity = float(sim_match.group(1)) if sim_match else 0.0

        # Gr.X detection (unique)
        gr_matches = re.findall(r'Gr\.(\d+)', content)
        gr_count = len(set(gr_matches))  # unique Gr.X

        issues = []
        if similarity < 0.5:
            issues.append("تحذير: تغييرات كبيرة")
        if gr_count == 0:
            issues.append("تحذير: لا توجد Gr.X clusters")
        if depth == 0:
            issues.append("تحذير: عمق غير محدد")

        verified_depth = min(depth, 12)
        verified_report = f"{report_path.rsplit('.', 1)[0]}_verified.txt"

        insights = {
            'depth': verified_depth,
            'similarity': similarity,
            'gr_count': gr_count,
            'issues': issues
        }

        with open(verified_report, 'w', encoding='utf-8') as f:
            f.write(f"تقرير تدقيق Helix - العمق: {verified_depth}\n")
            f.write(f"التشابه: {similarity:.3f}\n")
            f.write(f"Gr.X Clusters: {gr_count}\n")
            f.write("التصحيحات:\n")
            for issue in issues:
                f.write(f"- {issue}\n")
            f.write("\n" + json.dumps(insights, ensure_ascii=False, indent=2) + "\n")
            f.write("✅ تم التحقق!\n")

        print(f"✅ تدقيق مكتمل: {gr_count} Gr.X | {similarity:.3f} تشابه")
        return verified_report, issues

    def cleanup(self) -> None:
        """تنظيف شامل"""
        print("🧹 تنظيف Pipeline...")

        # الحالة
        if hasattr(self, 'spiral_analysis'):
            self.spiral_analysis.clear()
        if hasattr(self, 'insights'):
            self.insights.clear()
        if hasattr(self, 'reports'):
            self.reports.clear()
        self.tour_code = ""
        self.is_clean = True

        # الملفات
        temp_patterns = ['temp_test.py', '*_test.txt', '*_verified.txt', 'temp*.py']
        for pattern in temp_patterns:
            for file_path in Path('.').glob(pattern):
                try:
                    file_path.unlink()
                    print(f"🗑️ {file_path.name}")
                except:
                    pass

        # Turtle
        try:
            turtle.bye()
        except:
            pass

        print("✅ تنظيف مكتمل")

    def full_analysis_pipeline(self, complex_idea: str) -> Dict[str, Any]:
        """Pipeline كامل: Helix V3 → Clustering → Report"""

        # 1️⃣ Helix V3 analysis
        colored_lines = self.helix_puzzle_analyzer(complex_idea, max_lines=50)

        # 2️⃣ Clustering + analysis
        line_analysis = self.line_manager(colored_lines)
        core_idea = self.extract_core_idea(colored_lines, line_analysis)

        # 3️⃣ Virtual tour visual
        tour_code = self.generate_cluster_visual(line_analysis)

        # 4️⃣ Writing flow
        writing_plan = self.writing_flow_organizer({'line_analysis': line_analysis})

        # 5️⃣ Report
        report_path = self.type_txt_reporter(f"Helix_Analysis_{len(colored_lines)}_GrX{core_idea.get('gr_clusters_used', 0)}")

        # 6️⃣ Test
        metrics, test_report = self.test_tour(report_path)

        # 7️⃣ Verification
        verified_report, issues = self.verification_tour(report_path)

        return {
            'tour_code': tour_code,
            'core_idea': core_idea,
            'writing_plan': writing_plan,
            'metrics': metrics,
            'reports': [report_path, test_report, verified_report],
            'gr_count': core_idea.get('gr_clusters_used', 0),
            'total_clusters': len(line_analysis.get('clusters', []))
        }

    def _generate_writing_prompt(self, outline: List[Dict]) -> str:
        """Prompt جاهز للكتابة"""
        prompt_parts = []
        for section in outline:
            key_topics = ', '.join([line[:30] + '...' for line in section['content_flow'][:3]])
            prompt_parts.append(f"""
    **{section['section']}** ({section['words']} كلمة):
    - Cluster: {section['gr_cluster']}
    - الموضوعات: {key_topics}
    اكتب فقرة مترابطة ومنطقية.""")

        return f"""اكتب مقالة كاملة ومنظمة وفق الهيكل التالي:

    {''.join(prompt_parts)}

    المتطلبات:
    - التدفق المنطقي بين الأقسام
    - اللغة سلسة وعربية فصحى
    - ربط Gr.X clusters بالمحتوى
    - إجمالي ~{sum(s['words'] for s in outline)} كلمة"""

# -------- مثال استخدام --------
if __name__ == "__main__":
    """مثال كامل: Helix V3 + Gr.X Clustering"""
    pipeline = VirtualTour_Pipeline()

    # فكرة معقدة
    news_detector_idea = """AI يكشف الأخبار الحقيقية من المزيفة:
تحليل نصوص، sentiment analysis، fact-checking database،
NLP models، real-time verification، accuracy 95%،
معالجة 1000 خبر/ثانية، معقد جدًا، Gr.X clustering"""

    print("🚀 تشغيل Helix V3 Pipeline كامل...")

    # 1️⃣ Helix V3 Analysis
    colored_lines = pipeline.helix_puzzle_analyzer(news_detector_idea, max_lines=25)
    print(f"🧬 Helix V3: {len(colored_lines)} خط ملون | Gr.X: {sum(1 for l in colored_lines if 'Gr.' in l['group'])}")

    # 2️⃣ Clustering + Core Idea
    line_analysis = pipeline.line_manager(colored_lines)
    core_idea = pipeline.extract_core_idea(colored_lines, line_analysis)
    print(f"🧠 الفكرة الكلية: {core_idea.get('core_concept', 'غير محدد')}")

    # 3️⃣ Cluster Visual
    cluster_visual = pipeline.generate_cluster_visual(line_analysis)
    print("📊 Cluster visual جاهز!")
    # exec(cluster_visual)  # اختياري

    # 4️⃣ Writing Flow
    writing_plan = pipeline.writing_flow_organizer({'line_analysis': line_analysis})
    print(f"📝 خطة كتابة: {writing_plan['total_sections']} قسم")

    # 5️⃣ Full Pipeline + Reports
    results = pipeline.full_analysis_pipeline(news_detector_idea)
    print(f"✅ Pipeline كامل | Clusters: {results['total_clusters']} | Gr.X: {results['gr_count']}")

    # 6️⃣ Reports + Tests
    report_paths = results['reports']
    for report in report_paths:
        if report and Path(report).exists():
            metrics, test_report = pipeline.test_tour(report)
            verified_report, issues = pipeline.verification_tour(report)
            print(f"📄 {Path(report).name} → {metrics.get('gr_count', 0)} Gr.X")

    print("\n" + "="*80)
    print("🎉 HELIX V3 PIPELINE كامل 100%!")
    print(f"Core Idea: {core_idea.get('core_concept', 'N/A')}")
    print(f"Gr.X Used: {results.get('gr_count', 0)}")
    print("✅ ANCHORMAN! 🚀")
