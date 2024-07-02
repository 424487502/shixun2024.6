import os
from django.http import JsonResponse
import tempfile
from zhconv import convert
import whisper

def transcribe_audio(audio_file):
    # 创建临时文件对象
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        for chunk in audio_file.chunks():
            temp_audio.write(chunk)
        temp_audio.flush()
        temp_audio.close()
        file_path = temp_audio.name

        # 加载 Whisper 模型
        model = whisper.load_model("./model/base.pt")

        # 转录音频文件
        result = model.transcribe(file_path)

        # 将繁体中文转换为简体中文
        res = convert(result['text'], 'zh-cn')

        # 删除临时文件
        os.remove(file_path)

        return res