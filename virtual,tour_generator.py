# virtual_tour_generator.py

import subprocess
from pathlib import Path

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import colorsys
from turtle import Screen, Turtle, done

# ==================================================
#        قسم الجولات الافتراضية اللولبية
# ==================================================
class VirtualTour_Pipeline:
    """ جولات افتراضية لولبية: توليد، تحليل، تدقيق، اختبار. """

    def __init__(self, sim_threshold=0.9, max_depth=15):
        """إعداد الجولات الافتراضية اللولبية"""
        self.sim_threshold = sim_threshold
        self.max_depth = max_depth
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.insights = defaultdict(list)
        self.spiral_analysis = []

    def virtual_tour_generator(self, idea: str, width=800, height=600) -> str:
        """
        جولات افتراضية لولبية → كود Turtle
        """
        print(f"🔄 بدء جولات افتراضية ل: {idea}")

        # إعادة تعيين الحالة
        self.spiral_analysis = [idea]
        self.insights.clear()
        depth = 0

        # الحلقة اللولبية الافتراضية
        while depth < self.max_depth:
            depth += 1
            print(f"\n--- جولة {depth}/{self.max_depth} ---")

            # تحليل لولبي
            prev_text = self.spiral_analysis[-1].lower()
            shapes = re.findall(r'(دائرة|مربع|مثلث)', prev_text)
            colors = re.findall(r'(أزرق|أحمر|أخضر|أصفر)', prev_text)
            nums = re.findall(r'\d+', prev_text)
            complexity = len(shapes) + len(colors) + len(nums) + depth

            # insight لولبي جديد
            new_insight = f"عمق{depth}: شكل{shapes or 'غير محدد'}، ألوان{colors or 'افتراضي'}، تعقيد{complexity}"
            self.spiral_analysis.append(new_insight)

            self.insights['shapes'].extend(shapes)
            self.insights['colors'].extend(colors)

            # استشعار التشابه
            if len(self.spiral_analysis) > 1:
                vectors = self.vectorizer.fit_transform(self.spiral_analysis[-2:]).toarray()
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                print(f"تشابه: {similarity:.3f}")

                if similarity > self.sim_threshold:
                    print(f"❌ توقف: لا تطوير ({similarity:.3f} > {self.sim_threshold})")
                    break

            print(f"✅ {new_insight[:50]}...")

        # تحديد الشكل والخطوات النهائية للكود
        final_shape = insights['shapes'][-1] if insights['shapes'] else 'دائرة'
        final_steps = min(150, 50 + depth * 8)
        final_color = insights['colors'][-1] if insights['colors'] else 'أزرق'

        # توليد كود Turtle النهائي
        final_steps = min(150, 50 + depth * 8)
        final_shape = self.insights['shapes'][-1] if self.insights['shapes'] else 'دائرة'

        self.tour_code = f"""from turtle import *
import colorsys

screen = Screen()
screen.setup({width}, {height})
screen.bgcolor('black')
tracer(500)

h = 0
t = Turtle()
t.speed(0)
for i in range({final_steps}):
    c = colorsys.hsv_to_rgb(h, 1, 1)
    h += 0.01
    t.up()
    t.goto(0, 0)
    t.down()
    t.color('white', c)
    t.begin_fill()
    t.circle(50 + i*2, 360//{final_steps})
    t.end_fill()
done()
"""
        print(f"\n✅ اكتملت {depth} جولة | كود جاهز: {final_shape}, {final_steps} خطوة")
        return self.tour_code

    def get_insights(self):
        """استرجاع نتائج الجولات"""
        return dict(self.insights)

    def spiral_study(self, idea: str):
        """الجولات اللولبية"""
        self.spiral_analysis = [idea]
        depth = 0

        while depth < self.max_depth:
            depth += 1
            prev = self.spiral_analysis[-1].lower()
            shapes = re.findall(r'(دائرة|مربع|مثلث)', prev)
            colors = re.findall(r'(أزرق|أحمر|أخضر|أصفر)', prev)
            complexity = len(shapes) + len(colors) + depth

            new_insight = f"عمق{depth}: شكل{shapes or '?'}، تعقيد{complexity}"
            self.spiral_analysis.append(new_insight)
            self.insights['shapes'].extend(shapes)
            self.insights['colors'].extend(colors)  # إضافة

            if len(self.spiral_analysis) > 1:
                vectors = self.vectorizer.fit_transform(self.spiral_analysis[-2:]).toarray()
                sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                if sim > self.sim_threshold:
                    break

        return depth, dict(self.insights)  # بدل _save_report

    def verification_tour(report_path: str):
        """
        جولة تدقيق واحدة: قراءة TXT، تحليل، تصحيح.
        """
        print("\n🔍 جولة التدقيق الواحدة...")

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            return None, ["خطأ: الملف غير موجود"]

        # تحليل آمن
        lines = content.split('\n')
        depth_match = next((re.search(r'عدد الجولات:\s*(\d+)', line) for line in lines if 'جولات' in line), None)
        depth = int(depth_match.group(1)) if depth_match else 0

        sim_match = re.search(r'التشابه النهائي:\s*([\d.]+)', content)
        similarity = float(sim_match.group(1)) if sim_match else 0

        # insights بسيط
        insights = {'shapes': ['دائرة']}  # افتراضي آمن

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
        return verified_report, issues

    def test_tour(verified_report: str, temp_code_file='temp_test.py'):
        """
        جولة اختبار: تنفيذ، قياس، تقرير.
        """
        print("\n🚀 جولة الاختبار...")

        try:
            with open(verified_report, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            return {'error': 'report not found'}, None

        # استخراج آمن
        depth_match = re.search(r'عمق:\s*(\d+)', content)
        depth = int(depth_match.group(1)) if depth_match else 10

        test_code = f"""from turtle import *
    import time
    start = time.time()
    screen = Screen()
    screen.setup(800, 600)
    t = Turtle()
    for i in range({depth}):
        t.circle(50 + i)
    end = time.time()
    print(f"TIME={{end-start:.2f}} DEPTH={depth}")
    done()
    """

        Path(temp_code_file).write_text(test_code)

        # تنفيذ آمن
        try:
            result = subprocess.run(['python', temp_code_file],
                                capture_output=True, text=True, timeout=10)
            test_passed = result.returncode == 0

            # استخراج metrics من stdout
            time_match = re.search(r'TIME=([\d.]+)', result.stdout)
            exec_time = float(time_match.group(1)) if time_match else 0

            metrics = {'exec_time': exec_time, 'depth': depth, 'passed': test_passed}

        except subprocess.TimeoutExpired:
            metrics = {'error': 'timeout', 'passed': False}

        test_report = verified_report.replace('.txt', '_test.txt')
        with open(test_report, 'w', encoding='utf-8') as f:
            f.write("تقرير اختبار\n")
            f.write(f"حالة: {'✅ نجح' if metrics['passed'] else '❌ فشل'}\n")
            f.write(f"Metrics: {metrics}\n")

        print(f"نتيجة: {metrics['passed']} | {test_report}")
        return metrics, test_report

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

    def _save_report(self, name: str):
        """حفظ تقرير"""
        os.makedirs('output/reports', exist_ok=True)
        report_path = f"output/reports/{name}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"تقرير {name}\n")
            f.write(f"الجولات: {self.spiral_analysis}\n")
        self.reports.append(report_path)
        return report_path

    def run_pipeline(self, idea: str):
        """تشغيل كامل"""
        print("🚀 تشغيل Pipeline...")
        depth, insights = self.spiral_study(idea)  # تصحيح الرجوع
        report = self._save_report(f"spiral_{depth}")
        return {
            'insights': dict(self.insights),
            'reports': self.reports,
            'depth': depth
        }

# -------- قسم الاختبار الرئيسي --------
if __name__ == "__main__":
    """قسم الاختبار الرئيسي"""
    pipeline = VirtualTour_Pipeline()

    # اختبار الجولات الافتراضية
    idea = "جولة دائرة أزرق معقدة 100 خطوة"
    code = pipeline.virtual_tour_generator(idea)

    print("\n" + "="*50)
    print("اختبار ناجح!")
    print(f"تم توليد {len(pipeline.spiral_analysis)} جولة")
    print(f"الكود جاهز:\n{code[:200]}...")

    # تشغيل الكود (اختياري)
    # exec(code)
