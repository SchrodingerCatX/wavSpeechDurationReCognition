import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                           QVBoxLayout, QWidget, QFileDialog, QTextEdit, QMessageBox,
                           QProgressBar, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import os
import librosa
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from collections import Counter
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 添加音频特征表格窗口类
class AudioFeaturesWindow(QWidget):
    def __init__(self, features_data):
        super().__init__()
        self.features_data = features_data
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('音频特征分析')
        self.setGeometry(350, 350, 1000, 400)
        
        layout = QVBoxLayout()
        
        # 创建表格
        self.table = QTableWidget()
        self.table.setColumnCount(6)  # 增加一列显示平滑后的语言结果
        self.table.setHorizontalHeaderLabels(['频谱标准差(spec_std)', '音高标准差(pitch_std)', '节奏(tempo)', '谐波平均值(harmonic_mean)', '初步检测语言', '平滑后语言'])
        
        # 设置表格数据
        if self.features_data:
            self.table.setRowCount(len(self.features_data))
            for i, data in enumerate(self.features_data):
                self.table.setItem(i, 0, QTableWidgetItem(f"{data['spec_std']:.2f}"))
                self.table.setItem(i, 1, QTableWidgetItem(f"{data['pitch_std']:.2f}"))
                self.table.setItem(i, 2, QTableWidgetItem(f"{data['tempo']:.2f}"))
                self.table.setItem(i, 3, QTableWidgetItem(f"{data['harmonic_mean']:.4f}"))
                
                # 添加初步检测语言列
                language = data.get('language', 'unknown')
                # 将语言代码转换为更友好的显示名称
                language_display = {
                    'en': '英语',
                    'zh': '普通话',
                    'yue': '粤语',
                    'cmn': '普通话',
                    'unknown': '未知'
                }.get(language, language)
                self.table.setItem(i, 4, QTableWidgetItem(language_display))
                
                # 添加平滑后语言列
                smoothed_language = data.get('smoothed_language', 'unknown')
                # 将语言代码转换为更友好的显示名称
                smoothed_language_display = {
                    'en': '英语',
                    'zh': '普通话',
                    'yue': '粤语',
                    'cmn': '普通话',
                    'unknown': '未知'
                }.get(smoothed_language, smoothed_language)
                self.table.setItem(i, 5, QTableWidgetItem(smoothed_language_display))
        
        # 调整表格列宽
        header = self.table.horizontalHeader()
        for i in range(6):  # 更新为6列
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        
        layout.addWidget(self.table)
        self.setLayout(layout)

# 添加工作线程类
class LanguageDetectionWorker(QThread):
    # 定义信号
    progress_signal = pyqtSignal(int, str)  # 进度值和状态消息
    result_signal = pyqtSignal(dict)  # 处理结果
    error_signal = pyqtSignal(str)  # 错误信息
    features_signal = pyqtSignal(list)  # 音频特征数据
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.files = []
        self.segment_duration = 2.0
        self.overlap = 0.7
        self.model = None
        self.processor = None
        self.device = None
        self.language_mapping = {}
        self.audio_features = []
        
    def set_params(self, files, model, processor, device, language_mapping):
        self.files = files
        self.model = model
        self.processor = processor
        self.device = device
        self.language_mapping = language_mapping
        self.audio_features = []
        
    def load_audio(self, file_path):
        """加载音频文件并进行预处理"""
        try:
            # 使用librosa加载音频文件，并确保采样率和单声道
            audio, sr = librosa.load(file_path, sr=16000, mono=True, duration=None, res_type='kaiser_fast', offset=0.0, dtype=np.float32)
            
            # 确保音频数据是浮点型
            if audio.dtype != np.float32 and audio.dtype != np.float64:
                audio = audio.astype(np.float32)
            
            # 音频标准化
            audio = librosa.util.normalize(audio)
            
            return audio, sr
            
        except Exception as e:
            print(f"加载音频文件出错 {os.path.basename(file_path)}: {str(e)}")
            return None, 0
            
    def segment_audio(self, audio, sr, segment_duration=5.0, overlap=0.5):
        """将音频分割成固定时长的片段，使用指定的窗口大小和重叠比例"""
        # 计算每个片段的样本数
        segment_length = int(segment_duration * sr)
        hop_length = int(segment_length * (1 - overlap))
        
        segments = []
        segment_times = []
        
        # 如果音频总长度小于片段长度，直接返回整个音频
        if len(audio) <= segment_length:
            return [audio], [0.0]
        
        for start in range(0, len(audio) - segment_length + 1, hop_length):
            end = start + segment_length
            
            # 如果最后一个片段太短（小于1秒），则跳过
            if end - start < sr:
                continue
                
            segment = audio[start:end]
            segments.append(segment)
            segment_times.append(start / sr)
            
        # 处理最后一个片段（如果需要）
        last_start = start + hop_length
        if last_start < len(audio) and len(audio) - last_start >= sr:
            last_segment = audio[last_start:]
            # 如果最后一个片段太短，进行填充
            if len(last_segment) < segment_length:
                padding = segment_length - len(last_segment)
                last_segment = np.pad(last_segment, (0, padding), 'constant')
            segments.append(last_segment)
            segment_times.append(last_start / sr)
            
        return segments, segment_times
        
    def smooth_predictions(self, language_codes, confidences, window_size=3):
        """
        对预测结果进行平滑处理，减少噪声
        
        参数:
            language_codes: 语言代码列表
            confidences: 置信度列表
            window_size: 平滑窗口大小
            
        返回:
            smoothed_codes: 平滑后的语言代码列表
        """
        if len(language_codes) <= 1:
            return language_codes
        
        # 创建平滑后的结果列表
        smoothed_codes = language_codes.copy()
        
        print("before process:",smoothed_codes)

        # 语言偏好权重 - 平衡各语言权重
        lang_bias = {
            'en': 1.0,    # 提高英语权重
            'zh': 1.0,    # 保持普通话权重正常
            'yue': 1.0,   # 保持粤语权重正常
            'cmn': 1.0,   # 保持普通话权重正常
            'unknown': 0.5 # 降低未知语言权重
        }
        
        # 对每个位置应用平滑
        for k in range(len(language_codes)):
            # 确定窗口范围
            #对头尾进行处理
            i = 0
            if k==0:
                i=1
            elif k==len(language_codes)-1:
                i=len(language_codes)-2
            else:
                i=k

            start = max(0, i - window_size // 2)
            end = min(len(language_codes), i + window_size // 2 + 1)
            window = language_codes[start:end]
            window_confidences = confidences[start:end]
            
            print("start",start)
            print("end",end)
            print("window",window)

            # 统计窗口内各语言出现的次数，并考虑置信度和语言偏好
            lang_scores = {}
            for j, lang in enumerate(window):
                if lang not in lang_scores:
                    lang_scores[lang] = 0
                # 应用语言偏好权重
                bias = lang_bias.get(lang, 1.0)
                lang_scores[lang] += window_confidences[j] * bias
            
            print("lang_scores",lang_scores)

            # 选择得分最高的语言
            if lang_scores:
                # 如果同时存在英语，中文与粤语，取最开始的语言，因为认为语言切换时才会有突变值
                if 'en' in lang_scores and 'zh' in lang_scores and 'yue' in lang_scores:
                    smoothed_codes[k] = language_codes[k-1]
                else:
                    smoothed_codes[k] = max(lang_scores, key=lang_scores.get)

                # 如果英语和中文/粤语的得分非常接近（差距小于0.2），优先选择中文/粤语
                # if 'en' in lang_scores and ('zh' in lang_scores or 'yue' in lang_scores or 'cmn' in lang_scores):
                #     en_score = lang_scores['en']
                #     zh_score = lang_scores.get('zh', 0)
                #     yue_score = lang_scores.get('yue', 0)
                #     cmn_score = lang_scores.get('cmn', 0)
                    
                #     # 获取中文类语言的最高得分
                #     chinese_score = max(zh_score, yue_score, cmn_score)
                #     chinese_lang = 'zh' if zh_score == chinese_score else ('yue' if yue_score == chinese_score else 'cmn')
                    
                #     # 如果英语得分仅略高于中文，选择中文
                #     if en_score > chinese_score and en_score - chinese_score < 0.2:
                #         smoothed_codes[i] = chinese_lang
                #     else:
                #         smoothed_codes[i] = max(lang_scores, key=lang_scores.get)
                # else:
                #     smoothed_codes[i] = max(lang_scores, key=lang_scores.get)
        print("middle result_smoothed_codes",smoothed_codes)

        # 应用额外的平滑规则
        # 1. 单点平滑：检查相邻值，如果当前值与相邻值不同且前后相邻值相同，则修改当前值
        if len(smoothed_codes) > 2:
            for i in range(1, len(smoothed_codes) - 1):
                if (smoothed_codes[i] != smoothed_codes[i-1] and 
                    smoothed_codes[i] != smoothed_codes[i+1] and 
                    smoothed_codes[i-1] == smoothed_codes[i+1]):
                        smoothed_codes[i] = smoothed_codes[i-1]
        
        # 2. 双点平滑：检查连续两个相同值的前后邻居
        if len(smoothed_codes) > 6:  # 需要至少7个元素(2个当前值+前后各2个邻居值)
            for i in range(2, len(smoothed_codes) - 3):
                if (smoothed_codes[i] == smoothed_codes[i+1] and  # 当前两个值相同
                    smoothed_codes[i] != smoothed_codes[i-2] and  # 与前两个邻居不同
                    smoothed_codes[i] != smoothed_codes[i-1] and
                    smoothed_codes[i-2] == smoothed_codes[i-1] and  # 前两个邻居相同
                    smoothed_codes[i-2] == smoothed_codes[i+2] and  # 前后邻居相同
                    smoothed_codes[i-2] == smoothed_codes[i+3]):
                    smoothed_codes[i] = smoothed_codes[i-2]
                    smoothed_codes[i+1] = smoothed_codes[i-2]
        
        # 3. 三点平滑：检查连续三个相同值的前后邻居
        if len(smoothed_codes) > 8:  # 需要至少9个元素(3个当前值+前后各3个邻居值)
            for i in range(3, len(smoothed_codes) - 5):
                if (smoothed_codes[i] == smoothed_codes[i+1] and
                    smoothed_codes[i] == smoothed_codes[i+2] and  # 当前三个值相同
                    smoothed_codes[i] != smoothed_codes[i-3] and  # 与前三个邻居不同
                    smoothed_codes[i] != smoothed_codes[i-2] and
                    smoothed_codes[i] != smoothed_codes[i-1] and
                    smoothed_codes[i-3] == smoothed_codes[i-2] and  # 前三个邻居相同
                    smoothed_codes[i-2] == smoothed_codes[i-1] and
                    smoothed_codes[i-3] == smoothed_codes[i+3] and  # 前后邻居相同
                    smoothed_codes[i-3] == smoothed_codes[i+4] and
                    smoothed_codes[i-3] == smoothed_codes[i+5]):
                    smoothed_codes[i] = smoothed_codes[i-3]
                    smoothed_codes[i+1] = smoothed_codes[i-3]
                    smoothed_codes[i+2] = smoothed_codes[i-3]
        
        # 4. 通用N点平滑：查找连续相等的N个值，比较前后各N个邻居值
        # 如果前N个邻居值相等，后N个邻居值相等，前后邻居值不等，当前N个值与前后邻居值不同
        # 则将当前N个值修改成后面一个邻居值

        def apply_n_point_smoothing(n):
            if len(smoothed_codes) > n * 2 + (n-1):  # 需要至少 2n+n-1 个元素(n个当前值+前后各n个邻居值)
                for i in range(n, len(smoothed_codes) - (n*2-1)):
                    # 检查当前连续n个值是否相同
                    current_values_same = True
                    for j in range(1, n):
                        if smoothed_codes[i] != smoothed_codes[i+j]:
                            current_values_same = False
                            break
                    
                    if not current_values_same:
                        continue
                    
                    # 检查前n个邻居值是否相同
                    prev_neighbors_same = True
                    for j in range(1, n):
                        if smoothed_codes[i-n] != smoothed_codes[i-n+j]:
                            prev_neighbors_same = False
                            break
                    
                    if not prev_neighbors_same:
                        continue
                    
                    # 检查后n个邻居值是否相同
                    next_neighbors_same = True
                    for j in range(1, n):
                        if smoothed_codes[i+n] != smoothed_codes[i+n+j]:
                            next_neighbors_same = False
                            break
                    
                    if not next_neighbors_same:
                        continue
                    
                    # 检查前后邻居值是否不同
                    if smoothed_codes[i-n] == smoothed_codes[i+n]:
                        continue
                    
                    # 检查当前值与前后邻居值是否不同
                    if smoothed_codes[i] == smoothed_codes[i-n] or smoothed_codes[i] == smoothed_codes[i+n]:
                        continue
                    
                    # 满足所有条件，将当前n个值修改为后面邻居值
                    for j in range(n):
                        smoothed_codes[i+j] = smoothed_codes[i+n]
                    
                    print(f"N点平滑(N={n})应用于位置 {i}: 将连续{n}个值修改为后邻居值 [{smoothed_codes[i+n]}]")
        
        # 统计smoothed_codes中各语言出现的次数
        lang_counts = {}
        for lang in smoothed_codes:
            if lang not in lang_counts:
                lang_counts[lang] = 0
                lang_counts[lang] += 1
        
        # 找出出现次数最少的语言及其数量
        min_lang_count = float('inf')
        for lang, count in lang_counts.items():
            if count < min_lang_count:
                min_lang_count = count
        
        # 设置n的上限为出现次数最少的语言的数量，但不小于2
        max_n = max(2, min(min_lang_count, len(smoothed_codes)//3))
        print(f"各语言出现次数: {lang_counts}, 最少语言出现次数: {min_lang_count}, 设置n上限为: {max_n}")
        
        for n in range(2, max_n + 1):  # 调整上限为最少语言的出现次数
            print(f"n点平滑开始使用({n})..............")
            apply_n_point_smoothing(n)
        
        # 头部平滑处理：检查头部的前N个值
        # N的取值范围是1到出现次数最少语言的一半
        # head_max_n = max(1, min_lang_count // 2)
        # print(f"开始头部平滑处理，N上限为: {head_max_n}")
        
        # for n in range(1, head_max_n + 1):
        #     for i in range(min(n, len(smoothed_codes))):
        #         # 检查当前头部值是否与后续N个邻居值不同
        #         if i + n < len(smoothed_codes):
        #             neighbor_same = True
        #             neighbor_value = smoothed_codes[i + 1]
                    
        #             # 检查后续N个邻居值是否相同
        #             for j in range(1, n + 1):
        #                 if i + j >= len(smoothed_codes) or smoothed_codes[i + j] != neighbor_value:
        #                     neighbor_same = False
        #                     break
                    
        #             # 如果后续N个邻居值相同，且当前值与邻居值不同，则修改当前值
        #             if neighbor_same and smoothed_codes[i] != neighbor_value:
        #                 print(f"头部平滑(N={n})：位置 {i} 从 {smoothed_codes[i]} 修改为 {neighbor_value}")
        #                 smoothed_codes[i] = neighbor_value
        
        # 尾部平滑处理：检查尾部的后N个值
        # N的取值范围是1到出现次数最少语言的一半
        tail_max_n = max(1, min_lang_count // 2)
        print(f"开始尾部平滑处理，N上限为: {tail_max_n}")
        
        for n in range(1, tail_max_n + 1):
            for i in range(len(smoothed_codes) - 1, len(smoothed_codes) - min(n + 1, len(smoothed_codes)) - 1, -1):
                # 检查当前尾部值是否与前面N个邻居值不同
                if i - n >= 0:
                    neighbor_same = True
                    neighbor_value = smoothed_codes[i - 1]
                    
                    # 检查前面N个邻居值是否相同
                    for j in range(1, n + 1):
                        if i - j < 0 or smoothed_codes[i - j] != neighbor_value:
                            neighbor_same = False
                            break
                    
                    # 如果前面N个邻居值相同，且当前值与邻居值不同，则修改当前值
                    if neighbor_same and smoothed_codes[i] != neighbor_value:
                        print(f"尾部平滑(N={n})：位置 {i} 从 {smoothed_codes[i]} 修改为 {neighbor_value}")
                        smoothed_codes[i] = neighbor_value
        
        return smoothed_codes
        
    def process_audio_segment(self, audio_segment):
        """
        使用Whisper模型处理音频片段，识别语言
        
        参数:
            audio_segment: 音频片段数据
            
        返回:
            detected_language: 检测到的语言代码
            confidence: 置信度
        """
        try:
            # 将音频转换为Whisper模型所需的输入格式
            input_features = self.processor(audio_segment, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
            
            # 使用Whisper模型进行语言识别
            with torch.no_grad():
                # 首先检测语言
                result = self.model.detect_language(input_features)
                logits = result[0]
                
                # 确保logits是浮点类型，避免softmax错误
                if logits.dtype != torch.float32 and logits.dtype != torch.float64:
                    logits = logits.to(torch.float32)
                
                # 获取语言预测结果
                # 检查logits是否为0维张量
                if logits.dim() == 0:
                    print("begin 0 tensor process...........")
                    # 如果是0维张量，我们需要进行更复杂的分析
                    # 尝试从音频特征中提取更多信息来判断语言
                    # 提取音频的频谱特征
                    spec = librosa.feature.melspectrogram(y=audio_segment, sr=16000)
                    spec_mean = np.mean(spec)
                    spec_std = np.std(spec)
                    
                    # 提取音高特征，用于区分语言
                    pitch, _ = librosa.piptrack(y=audio_segment, sr=16000)
                    pitch_std = np.std(pitch[pitch > 0]) if np.any(pitch > 0) else 0
                    
                    # 提取节奏特征
                    onset_env = librosa.onset.onset_strength(y=audio_segment, sr=16000)
                    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=16000)
                    
                    # 提取谐波特征
                    harmonic, percussive = librosa.effects.hpss(audio_segment)
                    harmonic_mean = np.mean(harmonic)
                    #print("percussive.............",percussive)
                    
                    # 初始化detected_language变量，避免引用前未定义的错误
                    detected_language = "unknown"  # 默认值
                    
                    # 改进的启发式规则来区分语言 - 大幅提高英语识别准确率
                    # 打印调试信息
                    print(f"启发式分析 - spec_std: {float(spec_std):.2f}, pitch_std: {float(pitch_std):.2f}, tempo: {float(tempo):.2f}, harmonic_mean: {float(harmonic_mean):.4f}")

                    # 默认假设为英语
                    language_id = 0  # 英语
                    
                    # 只有当多个特征都强烈指向中文时，才判断为中文
                    # 大幅提高判断为中文的门槛
                    #if pitch_std > 1500 and harmonic_mean > 0.15 and spec_std < 12:
                        # 有强烈的中文特征时，进一步区分普通话和粤语
                    #浮点数的处理需要小心，可能需要根据情况调整，测试浮点数大小
                    print(abs(harmonic_mean) < 0.00005)
                    if abs(harmonic_mean) < 0.00005:
                        language_id = 1  # 普通话
                        detected_language = "zh"
                        print(f"判断为普通话: 高pitch_std={float(pitch_std):.2f}, 中等节奏tempo={float(tempo):.2f}")
                    else:
                        if pitch_std <1000:
                            language_id = 2  # 粤语
                            detected_language = "yue"
                            print(f"判断为粤语: 非常高的pitch_std={float(pitch_std):.2f}, 快速节奏tempo={float(tempo):.2f}")
                        else:
                            detected_language = "en"
                            print(f"判断为英语: pitch_std={float(pitch_std):.2f}不够高或谐波成分不足({float(harmonic_mean):.4f})或频谱变化大({float(spec_std):.2f})")
                    
                    #detected_language = self.model.config.id2label.get(language_id, 'en')
                    #print("通过en标签获取1次试试,detected_language",detected_language)

                    #if pitch_std <1000:
                    #    language_id = 2  # 粤语
                    #    detected_language = "yue"
                    #    print(f"判断为粤语: 非常高的pitch_std={float(pitch_std):.2f}, 快速节奏tempo={float(tempo):.2f}")
                    #elif pitch_std>1000 and pitch_std<1167:
                    #    language_id = 1  # 普通话
                    #    detected_language = "zh"
                    #    print(f"判断为普通话: 高pitch_std={float(pitch_std):.2f}, 中等节奏tempo={float(tempo):.2f}")
                    #else:
                        # 英语特征判断
                    #    detected_language = "en"
                    #    print(f"判断为英语: pitch_std={float(pitch_std):.2f}不够高或谐波成分不足({float(harmonic_mean):.4f})或频谱变化大({float(spec_std):.2f})")
                    
                    #detected_language = self.model.config.id2label.get(language_id, 'en')
                    print("detected_language",detected_language)

                    # 调整置信度 - 提高英语的置信度
                    if language_id == 0:  # 英语
                        confidence = 0.85  # 大幅提高英语置信度
                    elif language_id == 1:  # 普通话
                        confidence = 0.85  # 降低普通话置信度
                    elif language_id == 2:  # 粤语
                        confidence = 0.85  # 降低粤语置信度
                    else:
                        confidence = 0.50
                else:
                    print("multiply tensor process...........")
                    # 正常处理多维张量
                    # 应用语言偏好调整 - 更加平衡的权重分配
                    language_bias = {}
                    for i, lang in self.model.config.id2label.items():
                        if lang == 'zh':
                            language_bias[i] = 1.0  # 恢复普通话权重为正常值
                        elif lang == 'en':
                            language_bias[i] = 1.2  # 提高英语权重
                        elif lang == 'yue' or lang == 'zh-yue':
                            language_bias[i] = 1.0  # 恢复粤语权重为正常值
                        else:
                            language_bias[i] = 1.0  # 保持其他语言权重不变
                    
                    # 应用偏好权重
                    adjusted_logits = logits.clone()
                    for i, bias in language_bias.items():
                        if i < len(logits):
                            adjusted_logits[i] = logits[i] * bias
                    
                    # 使用调整后的logits获取语言ID
                    language_id = torch.argmax(adjusted_logits, dim=0).item()
                    detected_language = self.model.config.id2label[language_id]
                    
                    # 计算调整后的置信度
                    adjusted_probabilities = torch.nn.functional.softmax(adjusted_logits, dim=0)
                    confidence = adjusted_probabilities[language_id].item()
                
                # 打印检测结果，帮助调试
                print(f"Whisper检测语言: {detected_language}, 置信度: {confidence:.4f}")
                
                # 根据用户要求，不再使用转录和关键词检测方法
                # 直接返回模型的语言识别结果，但应用了权重调整
                
                # 将LABEL格式的语言代码映射到实际语言代码
                if detected_language.startswith('LABEL_'):
                    # 根据终端输出，模型可能将所有语言都识别为LABEL_0
                    # 我们需要更智能地判断实际语言
                    label_id = int(detected_language.split('_')[1])
                    print("label_id",label_id)
                    # 使用音频特征进一步判断语言类型
                    if label_id == 0:  # 如果模型认为是英语
                        # 提取额外特征来验证
                        spec = librosa.feature.melspectrogram(y=audio_segment, sr=16000)
                        spec_std = np.std(spec)
                        
                        # 提取音高特征
                        pitch, _ = librosa.piptrack(y=audio_segment, sr=16000)
                        pitch_std = np.std(pitch[pitch > 0]) if np.any(pitch > 0) else 0
                        
                        # 提取谐波特征
                        harmonic, percussive = librosa.effects.hpss(audio_segment)
                        harmonic_mean = np.mean(harmonic)
                        
                        # 大幅提高判断为中文的门槛，优先判断为英语
                        # 只有当pitch_std非常高且有其他明显中文特征时，才判断为中文
                        if pitch_std <1000:
                            detected_language = 'yue'  # 可能是粤语
                            print(f"特征分析判断为粤语: 非常高的pitch_std={float(pitch_std):.2f}, 快速节奏tempo={float(tempo):.2f}")
                        elif pitch_std>1000 and pitch_std<1167:
                            detected_language = 'zh'  # 更可能是普通话
                            print(f"特征分析判断为普通话: 高pitch_std={float(pitch_std):.2f}")
                        else:
                            # 默认判断为英语
                            detected_language = 'en'  
                            print(f"确认为英语: 特征不满足中文判断条件")
                    elif label_id == 1:
                        detected_language = 'zh'  # 普通话
                    elif label_id == 2:
                        detected_language = 'yue'  # 粤语
                    else:
                        detected_language = 'other'  # 其他语言
                    print(f"映射后的语言: {detected_language}")
                
                # 对特定语言的置信度进行微调
                if detected_language == 'zh':
                    # 保持普通话的置信度不变
                    confidence = min(confidence * 1.0, 0.85)
                elif detected_language == 'en':
                    # 提高英语的置信度
                    confidence = min(confidence * 1.0, 0.85)
                elif detected_language == 'yue':
                    # 保持粤语的置信度不变
                    confidence = min(confidence * 1.0, 0.85)
                
                # 存储音频特征数据，确保使用最终确定的语言
                # 提取音频的频谱特征
                spec = librosa.feature.melspectrogram(y=audio_segment, sr=16000)
                spec_std = np.std(spec)
                
                # 提取音高特征
                pitch, _ = librosa.piptrack(y=audio_segment, sr=16000)
                pitch_std = np.std(pitch[pitch > 0]) if np.any(pitch > 0) else 0
                
                # 提取节奏特征
                onset_env = librosa.onset.onset_strength(y=audio_segment, sr=16000)
                tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=16000)
                
                # 提取谐波特征
                harmonic, percussive = librosa.effects.hpss(audio_segment)
                harmonic_mean = np.mean(harmonic)
                
                # 将最终确定的语言结果添加到音频特征数据中
                self.audio_features.append({
                    'spec_std': float(spec_std),
                    'pitch_std': float(pitch_std),
                    'tempo': float(tempo),
                    'harmonic_mean': float(harmonic_mean),
                    'language': detected_language
                })
                
                return detected_language, confidence
                
        except Exception as e:
            print(f"处理音频片段时出错: {str(e)}")
            return "unknown", 0.0
    
    def run(self):
        try:
            # 初始化语言时长统计
            language_durations = {'英语': 0, '粤语': 0, '普通话': 0}
            total_files = len(self.files)
            total_progress = 0  # 总进度计数
            
            # 计算总处理量（文件数 * 每个文件的平均片段数）
            # 假设每个文件平均有20个片段，文件处理占20%，片段处理占80%
            file_weight = 20  # 每个文件的权重
            total_weight = total_files * file_weight
            
            # 处理每个文件
            for i, file_path in enumerate(self.files):
                # 更新文件处理进度（每个文件占总进度的一部分）
                file_progress = int((i / total_files) * 20)  # 文件处理占20%的进度
                self.progress_signal.emit(file_progress, f'正在处理: {os.path.basename(file_path)}')
                
                # 加载音频文件
                audio, sr = self.load_audio(file_path)
                if audio is None:
                    continue

                # 计算总时长
                total_duration = len(audio) / sr
                
                # 将音频分段
                segments, segment_times = self.segment_audio(audio, sr, self.segment_duration, self.overlap)

                # 如果没有有效片段，则跳过
                if not segments:
                    continue
                    
                # 处理每个片段
                segment_results = []
                language_codes = []
                confidences = []
                total_segments = len(segments)
                
                for j, (segment, start_time) in enumerate(zip(segments, segment_times)):
                    # 更新片段处理进度（片段处理占80%的进度）
                    segment_progress = int(20 + (i / total_files) * 80 + (j / total_segments) * (80 / total_files))
                    self.progress_signal.emit(segment_progress, 
                                             f'处理文件 {i+1}/{total_files}: {os.path.basename(file_path)} - 片段 {j+1}/{total_segments}')
                    
                    # 从片段中识别语言
                    detected_language, confidence = self.process_audio_segment(segment)
                    
                    # 保存结果
                    language_codes.append(detected_language)
                    confidences.append(confidence)
                    
                    # 计算片段时长
                    if j < len(segments) - 1:
                        duration = segment_times[j + 1] - start_time
                    else:
                        duration = total_duration - start_time
                    
                    # 保存结果
                    segment_results.append({
                        'start_time': start_time,
                        'duration': duration,
                        'language': detected_language,
                        'confidence': confidence
                    })
                
                # 对预测结果进行平滑处理
                smoothed_codes = self.smooth_predictions(language_codes, confidences)
                
                # 将平滑后的结果添加到音频特征数据中
                for j, smoothed_lang in enumerate(smoothed_codes):
                    if j < len(self.audio_features) - len(smoothed_codes) + j + 1:
                        self.audio_features[len(self.audio_features) - len(smoothed_codes) + j]['smoothed_language'] = smoothed_lang
                
                # 统计各语言片段数量
                language_counts = Counter(smoothed_codes)
                total_segments = len(smoothed_codes)
                
                # 计算各语言占比并更新时长统计
                for lang_code, count in language_counts.items():
                    if lang_code in self.language_mapping:
                        lang_name = self.language_mapping[lang_code]
                        lang_ratio = count / total_segments
                        lang_duration = lang_ratio * total_duration
                        language_durations[lang_name] += lang_duration
            
            # 发送进度完成信号
            self.progress_signal.emit(100, '处理完成')
            
            # 发送结果信号
            result = {
                'language_durations': language_durations,
                'files': self.files
            }
            self.result_signal.emit(result)
            
            # 发送音频特征信号
            self.features_signal.emit(self.audio_features)
            
        except Exception as e:
            self.error_signal.emit(str(e))


class LanguageDetectorWhisper(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_files = []
        print(torch.version.cuda)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_features = []  # 存储音频特征数据
        self.initUI()
        self.loadModel()
        
        # 创建工作线程和信号
        self.worker_thread = None
        self.progress_signal = None
        
    def loadModel(self):
        try:
            # 加载Whisper模型和处理器
            model_path = os.path.join(os.getcwd(), "models", "whisper-base")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Whisper模型未找到: {model_path}')
                
            self.processor = WhisperProcessor.from_pretrained(model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(self.device)
            
            # 设置语言映射
            self.language_mapping = {
                'en': '英语',
                'yue': '粤语',  # 粤语代码
                'zh': '普通话',  # 中文代码
                'cmn': '普通话',  # 普通话代码
                'LABEL_0': '英语',  # 添加LABEL格式的映射
                'LABEL_1': '普通话',
                'LABEL_2': '粤语',
                'LABEL_3': '其他语言'
            }
            
            # 设置语言检测映射
            self.detect_languages = {
                'en': 'english',
                'yue': 'cantonese',
                'zh': 'chinese',
                'cmn': 'chinese'
            }
            
            print(f"模型加载成功，使用设备: {self.device}")


        except Exception as e:
            QMessageBox.critical(self, "模型加载错误", f"无法加载Whisper模型: {str(e)}")
    
    def initUI(self):
        self.setWindowTitle('语言识别系统 (Whisper版)')
        self.setGeometry(300, 300, 700, 500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # 文件选择按钮
        self.file_btn = QPushButton('选择WAV文件', self)
        self.file_btn.clicked.connect(self.selectFiles)
        layout.addWidget(self.file_btn)
        
        # 识别按钮
        self.detect_btn = QPushButton('开始识别', self)
        self.detect_btn.clicked.connect(self.detectLanguages)
        self.detect_btn.setEnabled(False)
        layout.addWidget(self.detect_btn)
        
        # 添加音频特征按钮
        self.features_btn = QPushButton('查看音频特征', self)
        self.features_btn.clicked.connect(self.showAudioFeatures)
        self.features_btn.setEnabled(False)
        layout.addWidget(self.features_btn)

        # 进度条
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 进度标签
        self.progress_label = QLabel('', self)
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)

        # 结果显示区域
        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        central_widget.setLayout(layout)
        
    # 添加显示音频特征的方法
    def showAudioFeatures(self):
        if not self.audio_features:
            QMessageBox.information(self, "提示", "没有可用的音频特征数据，请先进行语言检测。")
            return
            
        self.features_window = AudioFeaturesWindow(self.audio_features)
        self.features_window.show()

    def selectFiles(self):
        files, _ = QFileDialog.getOpenFileNames(self, '选择WAV文件', '', 'WAV文件 (*.wav)')
        if files:
            self.selected_files = files
            self.detect_btn.setEnabled(True)
            self.file_btn.setText(f'已选择: {len(files)}个文件')
    
    def load_audio(self, file_path):
        """加载音频文件并进行预处理"""
        try:
            print(f"加载文件: {os.path.basename(file_path)}")
            
            # 使用librosa加载音频文件，并确保采样率和单声道
            try:
                audio, sr = librosa.load(file_path, sr=16000, mono=True, duration=None, res_type='kaiser_fast', offset=0.0, dtype=np.float32)
                print(f"音频采样率: {sr}, 音频形状: {audio.shape}, 时长: {len(audio)/sr:.2f}秒")
                
                # 确保音频数据是浮点型
                if audio.dtype != np.float32 and audio.dtype != np.float64:
                    audio = audio.astype(np.float32)
                
                # 对音频进行预处理，但保持原始时长
                # 1. 音频标准化 - 使音量一致，但不改变时长
                audio = librosa.util.normalize(audio)
                
                # 注释掉去除静音部分的代码，保持原始音频时长
                # non_silent_intervals = librosa.effects.split(audio, top_db=30)
                # if len(non_silent_intervals) > 0:
                #     non_silent_audio = []
                #     for interval in non_silent_intervals:
                #         non_silent_audio.extend(audio[interval[0]:interval[1]])
                #     if len(non_silent_audio) > 0:
                #         audio = np.array(non_silent_audio)
                
                # 注释掉预加重滤波器，避免改变音频特性
                # audio = librosa.effects.preemphasis(audio)
                
                print(f"预处理后音频形状: {audio.shape}, 时长: {len(audio)/sr:.2f}秒")
                    
                return audio, sr
            except Exception as e:
                print(f"使用librosa加载音频文件失败: {str(e)}")
                return None, 0
            
        except Exception as e:
            print(f"加载音频文件出错 {os.path.basename(file_path)}: {str(e)}")
            return None, 0
    
    def segment_audio(self, audio, sr, segment_duration=5.0, overlap=0.5):
        """
        将音频分割成固定时长的片段，使用指定的窗口大小和重叠比例
        
        参数:
            audio: 音频数据
            sr: 采样率
            segment_duration: 每个片段的时长（秒）
            overlap: 重叠比例
            
        返回:
            segments: 音频片段列表
            segment_times: 每个片段的开始时间列表
        """
        # 计算每个片段的样本数
        segment_length = int(segment_duration * sr)
        hop_length = int(segment_length * (1 - overlap))
        
        segments = []
        segment_times = []
        
        # 如果音频总长度小于片段长度，直接返回整个音频
        if len(audio) <= segment_length:
            return [audio], [0.0]
        
        for start in range(0, len(audio) - segment_length + 1, hop_length):
            end = start + segment_length
            
            # 如果最后一个片段太短（小于1秒），则跳过
            if end - start < sr:
                continue
                
            segment = audio[start:end]
            segments.append(segment)
            segment_times.append(start / sr)
            
        # 处理最后一个片段（如果需要）
        last_start = start + hop_length
        if last_start < len(audio) and len(audio) - last_start >= sr:
            last_segment = audio[last_start:]
            # 如果最后一个片段太短，进行填充
            if len(last_segment) < segment_length:
                padding = segment_length - len(last_segment)
                last_segment = np.pad(last_segment, (0, padding), 'constant')
            segments.append(last_segment)
            segment_times.append(last_start / sr)
            
        return segments, segment_times
    
    
    @pyqtSlot(int, str)
    def update_progress(self, value, message):
        """更新进度条和状态消息"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)        
    @pyqtSlot(dict)
    def process_results(self, results):
        """处理并显示结果"""
        language_durations = results['language_durations']
        files = results['files']
        
        # 显示结果
        result_text = "检测结果:\n\n"
        result_text += "语言时长统计 (基于片段占比计算):\n"
        for lang, duration in language_durations.items():
            result_text += f"{lang}: {duration:.2f} 秒\n"
        
        result_text += "\n处理的文件:\n"
        for file_path in files:
            result_text += f"- {os.path.basename(file_path)}\n"
        
        self.result_text.setText(result_text)
        
        # 启用音频特征按钮
        self.features_btn.setEnabled(True)
        
    @pyqtSlot(str)
    def handle_error(self, error_message):
        """处理错误"""
        QMessageBox.critical(self, "错误", f"处理过程中出现错误: {error_message}")
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
    @pyqtSlot(list)
    def update_features(self, features):
        """更新音频特征数据"""
        self.audio_features = features
    
    def detectLanguages(self):
        if not self.selected_files:
            return
            
        try:
            # 重置并显示进度指示器
            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_label.setText('准备处理...')
            
            # 清空之前的音频特征数据
            self.audio_features = []
            
            # 禁用识别按钮，防止重复点击
            self.detect_btn.setEnabled(False)
            
            # 创建并配置工作线程
            self.worker_thread = LanguageDetectionWorker()
            self.worker_thread.set_params(
                self.selected_files,
                self.model,
                self.processor,
                self.device,
                self.language_mapping
            )
            
            # 连接信号和槽
            self.worker_thread.progress_signal.connect(self.update_progress)
            self.worker_thread.result_signal.connect(self.process_results)
            self.worker_thread.error_signal.connect(self.handle_error)
            self.worker_thread.features_signal.connect(self.update_features)
            
            # 线程完成后重新启用按钮
            self.worker_thread.finished.connect(lambda: self.detect_btn.setEnabled(True))
            
            # 启动工作线程
            self.worker_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动处理线程时出错: {str(e)}")
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)
            self.detect_btn.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    ex = LanguageDetectorWhisper()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
